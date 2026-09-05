from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from src.model.state import RWKVState, RWKVStateSpec


def test_state_clone_has_no_storage_alias_and_detach_is_explicit() -> None:
    spec = RWKVStateSpec(n_layer=2, n_embd=8, n_head=2, head_size=4, batch_size=1)
    state = RWKVState.zeros(spec, device="cpu", dtype=torch.bfloat16)
    state.layers[0].time_mix_previous_x.requires_grad_(True)

    attached = state.clone()
    detached = state.detached()
    assert attached.layers[0].time_mix_previous_x.grad_fn is not None
    assert detached.layers[0].time_mix_previous_x.grad_fn is None
    for source, copied in zip(state.layers, attached.layers):
        assert source.time_mix_previous_x.data_ptr() != copied.time_mix_previous_x.data_ptr()
        assert source.wkv_matrix.data_ptr() != copied.wkv_matrix.data_ptr()
        assert source.channel_mix_previous_x.data_ptr() != copied.channel_mix_previous_x.data_ptr()

    attached.layers[0].time_mix_previous_x.data.add_(1)
    assert torch.count_nonzero(state.layers[0].time_mix_previous_x) == 0


def test_state_save_load_round_trip(tmp_path) -> None:
    spec = RWKVStateSpec(n_layer=2, n_embd=8, n_head=2, head_size=4, batch_size=2)
    state = RWKVState.zeros(spec, device="cpu", dtype=torch.bfloat16)
    state.layers[1].wkv_matrix.add_(torch.randn_like(state.layers[1].wkv_matrix))
    path = tmp_path / "state.pt"
    state.save(path)
    restored = RWKVState.load(path)

    assert restored.spec == state.spec
    assert restored.schema_version == state.schema_version
    assert restored.nbytes == state.nbytes
    for source, actual in zip(state.layers, restored.layers):
        assert torch.equal(source.time_mix_previous_x, actual.time_mix_previous_x)
        assert torch.equal(source.wkv_matrix, actual.wkv_matrix)
        assert torch.equal(source.channel_mix_previous_x, actual.channel_mix_previous_x)


def test_state_validation_rejects_shape_dtype_and_schema() -> None:
    spec = RWKVStateSpec(n_layer=1, n_embd=8, n_head=2, head_size=4, batch_size=1)
    state = RWKVState.zeros(spec, device="cpu", dtype=torch.bfloat16)
    layer = state.layers[0]
    with pytest.raises(ValueError, match="TimeMix"):
        RWKVState(
            spec,
            (replace(layer, time_mix_previous_x=torch.zeros(1, 7)),),
        )
    with pytest.raises(TypeError, match="float32"):
        RWKVState(spec, (replace(layer, wkv_matrix=layer.wkv_matrix.bfloat16()),))
    with pytest.raises(TypeError, match="activation dtype"):
        RWKVState(
            spec,
            (
                replace(
                    layer,
                    time_mix_previous_x=layer.time_mix_previous_x.to(torch.int32),
                    channel_mix_previous_x=layer.channel_mix_previous_x.to(torch.int32),
                ),
            ),
        )
    with pytest.raises(ValueError, match="devices differ"):
        RWKVState(
            spec,
            (replace(layer, channel_mix_previous_x=layer.channel_mix_previous_x.to("meta")),),
        )
    with pytest.raises(ValueError, match="schema"):
        RWKVState(spec, (layer,), schema_version=99)


def test_state_spec_rejects_inconsistent_head_shape() -> None:
    with pytest.raises(ValueError, match="n_head"):
        RWKVStateSpec(n_layer=1, n_embd=9, n_head=2, head_size=4, batch_size=1)


def test_state_load_rejects_unknown_file_keys(tmp_path) -> None:
    path = tmp_path / "invalid-state.pt"
    torch.save(
        {
            "schema_version": 1,
            "spec": {},
            "tensors": {},
            "unexpected": True,
        },
        path,
    )
    with pytest.raises(ValueError, match="keys"):
        RWKVState.load(path)
