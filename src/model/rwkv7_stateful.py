"""Differentiable RWKV-7 window continuation using the state-passing CUDA op."""

from __future__ import annotations

import importlib
from typing import Any

from .state import RWKVLayerState, RWKVState, RWKVStateSpec

CHUNK_LEN = 16


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for stateful RWKV execution") from error
    return torch


def state_spec_from_model(network: Any, batch_size: int) -> RWKVStateSpec:
    args = network.args
    head_size = int(args.head_size)
    n_head = int(args.dim_att) // head_size
    if int(args.dim_att) != int(args.n_embd):
        raise ValueError("the current stateful prototype requires dim_att == n_embd")
    return RWKVStateSpec(
        n_layer=int(args.n_layer),
        n_embd=int(args.n_embd),
        n_head=n_head,
        head_size=head_size,
        batch_size=batch_size,
    )


def shifted_difference(x: Any, previous_x: Any, sequence_start_mask: Any):
    torch = _torch()
    if tuple(previous_x.shape) != (x.shape[0], x.shape[2]):
        raise ValueError("previous-x must have shape [B, C]")
    if tuple(sequence_start_mask.shape) != (x.shape[0], x.shape[1]):
        raise ValueError("sequence_start_mask must have shape [B, T]")
    shifted = torch.cat((previous_x.unsqueeze(1), x[:, :-1]), dim=1)
    shifted = shifted.masked_fill(sequence_start_mask.bool().unsqueeze(-1), 0.0)
    return shifted - x, x[:, -1]


def _state_passing_operation():
    torch = _torch()

    class StatePassing(torch.autograd.Function):
        @staticmethod
        def forward(ctx, s0, r, w, k, v, a, b, starts):
            batch, timesteps, heads, head_size = r.shape
            if timesteps % CHUNK_LEN:
                raise ValueError("stateful windows must be divisible by 16")
            y = torch.empty_like(v)
            s_t = torch.empty_like(s0)
            checkpoints = torch.empty(
                batch,
                heads,
                timesteps // CHUNK_LEN,
                head_size,
                head_size,
                device=r.device,
                dtype=torch.float32,
            )
            state_a = torch.empty(
                batch,
                timesteps,
                heads,
                head_size,
                device=r.device,
                dtype=torch.float32,
            )
            torch.ops.rwkv7_statepassing_clampw.forward(
                s0, r, w, k, v, a, b, starts, y, s_t, checkpoints, state_a
            )
            ctx.save_for_backward(s0, r, w, k, v, a, b, starts, checkpoints, state_a)
            return y, s_t

        @staticmethod
        def backward(ctx, grad_y, grad_s_t):
            s0, r, w, k, v, a, b, starts, checkpoints, state_a = ctx.saved_tensors
            grad_s0 = torch.empty_like(s0)
            token_grads = [torch.empty_like(value) for value in (r, w, k, v, a, b)]
            torch.ops.rwkv7_statepassing_clampw.backward(
                s0,
                r,
                w,
                k,
                v,
                a,
                b,
                starts,
                grad_y.contiguous(),
                grad_s_t.contiguous(),
                checkpoints,
                state_a,
                grad_s0,
                *token_grads,
            )
            return grad_s0, *token_grads, None

    return StatePassing


def state_passing_reference(r, w, k, v, a, b, s0, sequence_start_mask):
    """Differentiable RWKV-7 recurrence for short or non-CUDA windows."""
    torch = _torch()
    values = (r, w, k, v, a, b)
    if any(value.shape != r.shape for value in values):
        raise ValueError("all WKV token tensors must have the same shape")
    if r.ndim != 3 or s0.ndim != 4:
        raise ValueError("WKV tokens and state must have rank 3 and 4")
    batch, timesteps, channels = r.shape
    if timesteps <= 0:
        raise ValueError("WKV recurrence requires at least one timestep")
    heads, head_size = s0.shape[1:3]
    if tuple(s0.shape) != (batch, heads, head_size, head_size):
        raise ValueError("WKV state must have shape [B, H, N, N]")
    if channels != heads * head_size:
        raise ValueError("WKV token channels do not match state heads")
    if tuple(sequence_start_mask.shape) != (batch, timesteps):
        raise ValueError("sequence_start_mask must have shape [B, T]")
    if s0.dtype != torch.float32:
        raise TypeError("WKV matrix state must be float32")
    if any(value.device != s0.device for value in values):
        raise ValueError("WKV tokens and state must use the same device")

    shaped = [value.view(batch, timesteps, heads, head_size) for value in values]
    r4, w4, k4, v4, a4, b4 = shaped
    w_decay = torch.exp(-torch.exp(-torch.nn.functional.softplus(-w4.float()) - 0.5))
    state = s0
    outputs = []
    for timestep in range(timesteps):
        reset = sequence_start_mask[:, timestep].bool().view(batch, 1, 1, 1)
        state = state.masked_fill(reset, 0.0)
        rr = r4[:, timestep].float()
        kk = k4[:, timestep].float()
        vv = v4[:, timestep].float()
        aa = a4[:, timestep].float()
        bb = b4[:, timestep].float()
        state_a = torch.einsum("bhik,bhk,bhj->bhij", state, aa, bb)
        state = (
            state * w_decay[:, timestep, :, None, :]
            + state_a
            + torch.einsum("bhj,bhi->bhij", kk, vv)
        )
        output = torch.einsum("bhj,bhij->bhi", rr, state)
        outputs.append(output.to(r.dtype))
    return torch.stack(outputs, dim=1).reshape(batch, timesteps, channels), state


def state_passing(r, w, k, v, a, b, s0, sequence_start_mask):
    torch = _torch()
    batch, timesteps, channels = r.shape
    if r.is_cuda and timesteps % CHUNK_LEN and not torch.is_grad_enabled():
        return state_passing_inference(r, w, k, v, a, b, s0, sequence_start_mask)
    if not r.is_cuda or timesteps % CHUNK_LEN:
        return state_passing_reference(r, w, k, v, a, b, s0, sequence_start_mask)
    head_size = s0.shape[-1]
    heads = channels // head_size
    values = [
        value.contiguous().view(batch, timesteps, heads, head_size) for value in (r, w, k, v, a, b)
    ]
    operation = _state_passing_operation()
    output, final_state = operation.apply(
        s0.contiguous(), *values, sequence_start_mask.contiguous()
    )
    return output.view(batch, timesteps, channels), final_state


def state_passing_inference(r, w, k, v, a, b, s0, sequence_start_mask):
    """Use the official forward recurrence for any T, never its aligned backward.

    The pinned CUDA forward loops over T and stores floor(T/16) checkpoints.
    Its backward requires aligned T, so this entry is strictly inference-only.
    No fake tokens are appended and no recurrent state update is discarded.
    """
    torch = _torch()
    if torch.is_grad_enabled():
        raise RuntimeError("unaligned CUDA forward is inference-only")
    if r.ndim != 3 or not r.is_cuda or r.dtype != torch.bfloat16:
        raise ValueError("expected BF16 CUDA tokens of shape [B,T,C]")
    batch, timesteps, channels = r.shape
    if timesteps <= 0 or channels % 64:
        raise ValueError("invalid inference WKV shape")
    heads = channels // 64
    if tuple(s0.shape) != (batch, heads, 64, 64) or s0.dtype != torch.float32:
        raise ValueError("expected FP32 WKV state [B,H,64,64]")
    if (
        tuple(sequence_start_mask.shape) != (batch, timesteps)
        or sequence_start_mask.dtype != torch.uint8
    ):
        raise ValueError("invalid inference sequence_start_mask")
    if any(
        value.shape != r.shape or value.dtype != r.dtype or value.device != r.device
        for value in (w, k, v, a, b)
    ):
        raise ValueError("inference WKV token tensors disagree")
    if s0.device != r.device or sequence_start_mask.device != r.device:
        raise ValueError("inference WKV device mismatch")
    values = [value.contiguous().view(batch, timesteps, heads, 64) for value in (r, w, k, v, a, b)]
    output = torch.empty_like(values[0])
    final_state = torch.empty_like(s0)
    checkpoints = torch.empty(
        batch, heads, timesteps // CHUNK_LEN, 64, 64, device=r.device, dtype=torch.float32
    )
    state_a = torch.empty(batch, timesteps, heads, 64, device=r.device, dtype=torch.float32)
    torch.ops.rwkv7_statepassing_clampw.forward(
        s0.contiguous(),
        *values,
        sequence_start_mask.contiguous(),
        output,
        final_state,
        checkpoints,
        state_a,
    )
    return output.view(batch, timesteps, channels), final_state


def time_mix_forward(
    layer: Any,
    x: Any,
    v_first: Any,
    state: RWKVLayerState,
    sequence_start_mask: Any,
    model_module: Any,
):
    batch, timesteps, channels = x.shape
    heads = layer.n_head
    difference, final_previous_x = shifted_difference(
        x, state.time_mix_previous_x, sequence_start_mask
    )
    xr = x + difference * layer.x_r
    xw = x + difference * layer.x_w
    xk = x + difference * layer.x_k
    xv = x + difference * layer.x_v
    xa = x + difference * layer.x_a
    xg = x + difference * layer.x_g

    r = layer.receptance(xr)
    w = layer.w0 + model_module.torch.tanh(xw @ layer.w1) @ layer.w2
    k = layer.key(xk)
    v = layer.value(xv)
    if layer.layer_id == 0:
        v_first = v
    else:
        v12 = (xv @ layer.v1) @ layer.v2
        v = model_module.tmix_vres_gate_bf16_v3(v, v_first, layer.v0, v12)
    a = model_module.tmix_a_gate_bf16(layer.a0, (xa @ layer.a1) @ layer.a2)
    g = model_module.torch.sigmoid(xg @ layer.g1) @ layer.g2
    k, negative_kk, kka = model_module.tmix_kk_pre_bf16_v5(
        k, layer.k_k.view(-1), a, layer.k_a.view(-1)
    )
    mixed, final_wkv = state_passing(
        r,
        w,
        k,
        v,
        negative_kk,
        kka,
        state.wkv_matrix,
        sequence_start_mask,
    )
    mixed = model_module.tmix_lnx_rkvres_xg_bf16_v1(
        mixed,
        r,
        k,
        v,
        layer.r_k,
        layer.ln_x.weight,
        layer.ln_x.bias,
        g,
    )
    output = layer.output(mixed)
    next_state = RWKVLayerState(
        time_mix_previous_x=final_previous_x,
        wkv_matrix=final_wkv,
        channel_mix_previous_x=state.channel_mix_previous_x,
    )
    if output.shape != (batch, timesteps, channels) or heads * layer.head_size != channels:
        raise ValueError("TimeMix output shape mismatch")
    return output, v_first, next_state


def channel_mix_forward(layer: Any, x: Any, state: RWKVLayerState, sequence_start_mask: Any):
    torch = _torch()
    difference, final_previous_x = shifted_difference(
        x, state.channel_mix_previous_x, sequence_start_mask
    )
    key = x + difference * layer.x_k
    key = torch.relu(layer.key(key)) ** 2
    output = layer.value(key)
    return output, RWKVLayerState(
        time_mix_previous_x=state.time_mix_previous_x,
        wkv_matrix=state.wkv_matrix,
        channel_mix_previous_x=final_previous_x,
    )


def stateful_forward_embeddings(
    network: Any,
    embeddings: Any,
    state: RWKVState | None = None,
    sequence_start_mask: Any | None = None,
    *,
    detach_state: bool = False,
):
    torch = _torch()
    batch, timesteps, _ = embeddings.shape
    if timesteps <= 0:
        raise ValueError("stateful forward requires at least one timestep")
    spec = state_spec_from_model(network, batch)
    if state is None:
        state = RWKVState.zeros(spec, device=embeddings.device, dtype=embeddings.dtype)
    elif state.spec != spec:
        raise ValueError("provided state spec does not match model and batch")
    if sequence_start_mask is None:
        sequence_start_mask = torch.zeros(
            batch, timesteps, device=embeddings.device, dtype=torch.uint8
        )
    if sequence_start_mask.dtype != torch.uint8 or not sequence_start_mask.is_contiguous():
        raise TypeError("sequence_start_mask must be contiguous uint8")
    if int(network.args.grad_cp) != 0:
        raise ValueError("stateful prototype does not yet support activation checkpointing")

    x = embeddings
    v_first = torch.empty_like(x)
    next_layers = []
    model_module = importlib.import_module(network.__class__.__module__)
    for index, block in enumerate(network.blocks):
        if index == 0:
            x = block.ln0(x)
        normalized = block.ln1(x)
        attention, v_first, attention_state = time_mix_forward(
            block.att,
            normalized,
            v_first,
            state.layers[index],
            sequence_start_mask,
            model_module,
        )
        x = x + attention
        normalized = block.ln2(x)
        channel, layer_state = channel_mix_forward(
            block.ffn, normalized, attention_state, sequence_start_mask
        )
        x = x + channel
        next_layers.append(layer_state)
    hidden = network.ln_out(x)
    next_state = RWKVState(spec=spec, layers=tuple(next_layers))
    if detach_state:
        next_state = next_state.detached()
    return hidden, next_state


def stateful_forward(
    network: Any,
    input_ids: Any,
    state: RWKVState | None = None,
    sequence_start_mask: Any | None = None,
    *,
    detach_state: bool = False,
    return_logits: bool = True,
):
    embeddings = network.emb(input_ids)
    hidden, next_state = stateful_forward_embeddings(
        network,
        embeddings,
        state,
        sequence_start_mask,
        detach_state=detach_state,
    )
    return (network.head(hidden) if return_logits else hidden), next_state
