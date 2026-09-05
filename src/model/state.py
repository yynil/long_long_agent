"""Versioned complete recurrent state for RWKV-7 windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STATE_SCHEMA_VERSION = 1


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for RWKV state operations") from error
    return torch


@dataclass(frozen=True)
class RWKVStateSpec:
    n_layer: int
    n_embd: int
    n_head: int
    head_size: int
    batch_size: int

    def __post_init__(self) -> None:
        if min(self.n_layer, self.n_embd, self.n_head, self.head_size, self.batch_size) <= 0:
            raise ValueError("all RWKV state dimensions must be positive")
        if self.n_head * self.head_size != self.n_embd:
            raise ValueError("n_head * head_size must equal n_embd")


@dataclass(frozen=True)
class RWKVLayerState:
    time_mix_previous_x: Any
    wkv_matrix: Any
    channel_mix_previous_x: Any

    def clone(self, *, detach: bool = False) -> RWKVLayerState:
        def copy(tensor):
            if detach:
                tensor = tensor.detach()
            return tensor.clone()

        return RWKVLayerState(
            time_mix_previous_x=copy(self.time_mix_previous_x),
            wkv_matrix=copy(self.wkv_matrix),
            channel_mix_previous_x=copy(self.channel_mix_previous_x),
        )


@dataclass(frozen=True)
class RWKVState:
    spec: RWKVStateSpec
    layers: tuple[RWKVLayerState, ...]
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def zeros(cls, spec: RWKVStateSpec, *, device: Any, dtype: Any) -> RWKVState:
        torch = _torch()
        layers = tuple(
            RWKVLayerState(
                time_mix_previous_x=torch.zeros(
                    spec.batch_size, spec.n_embd, device=device, dtype=dtype
                ),
                wkv_matrix=torch.zeros(
                    spec.batch_size,
                    spec.n_head,
                    spec.head_size,
                    spec.head_size,
                    device=device,
                    dtype=torch.float32,
                ),
                channel_mix_previous_x=torch.zeros(
                    spec.batch_size, spec.n_embd, device=device, dtype=dtype
                ),
            )
            for _ in range(spec.n_layer)
        )
        return cls(spec=spec, layers=layers)

    def validate(self) -> None:
        torch = _torch()
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported RWKV state schema version: {self.schema_version}")
        if len(self.layers) != self.spec.n_layer:
            raise ValueError("state layer count does not match spec")
        expected_x = (self.spec.batch_size, self.spec.n_embd)
        expected_wkv = (
            self.spec.batch_size,
            self.spec.n_head,
            self.spec.head_size,
            self.spec.head_size,
        )
        device = None
        x_dtype = None
        for index, layer in enumerate(self.layers):
            tensors = (
                layer.time_mix_previous_x,
                layer.wkv_matrix,
                layer.channel_mix_previous_x,
            )
            if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
                raise TypeError(f"layer {index} contains a non-tensor state")
            if tuple(layer.time_mix_previous_x.shape) != expected_x:
                raise ValueError(f"layer {index} TimeMix previous-x shape mismatch")
            if tuple(layer.channel_mix_previous_x.shape) != expected_x:
                raise ValueError(f"layer {index} ChannelMix previous-x shape mismatch")
            if tuple(layer.wkv_matrix.shape) != expected_wkv:
                raise ValueError(f"layer {index} WKV matrix shape mismatch")
            if layer.wkv_matrix.dtype != torch.float32:
                raise TypeError(f"layer {index} WKV matrix must be float32")
            if layer.time_mix_previous_x.dtype != layer.channel_mix_previous_x.dtype:
                raise TypeError(f"layer {index} previous-x dtypes differ")
            if layer.time_mix_previous_x.dtype not in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
            }:
                raise TypeError(f"layer {index} previous-x must use a floating activation dtype")
            if len({tensor.device for tensor in tensors}) != 1:
                raise ValueError(f"layer {index} state devices differ")
            if device is None:
                device = layer.wkv_matrix.device
                x_dtype = layer.time_mix_previous_x.dtype
            elif layer.wkv_matrix.device != device or layer.time_mix_previous_x.dtype != x_dtype:
                raise ValueError("all state layers must share device and previous-x dtype")

    @property
    def device(self):
        return self.layers[0].wkv_matrix.device

    @property
    def previous_x_dtype(self):
        return self.layers[0].time_mix_previous_x.dtype

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in self.layers
            for tensor in (
                layer.time_mix_previous_x,
                layer.wkv_matrix,
                layer.channel_mix_previous_x,
            )
        )

    def tensors(self) -> tuple[Any, ...]:
        return tuple(
            tensor
            for layer in self.layers
            for tensor in (
                layer.time_mix_previous_x,
                layer.wkv_matrix,
                layer.channel_mix_previous_x,
            )
        )

    def shares_storage_with(self, other: RWKVState) -> bool:
        if not isinstance(other, RWKVState):
            raise TypeError("storage alias checks require another RWKVState")

        def storage_key(tensor):
            return tensor.device, tensor.untyped_storage().data_ptr()

        own_storage = {storage_key(tensor) for tensor in self.tensors()}
        return any(storage_key(tensor) in own_storage for tensor in other.tensors())

    def clone(self, *, detach: bool = False) -> RWKVState:
        """Deep-copy every tensor; detach is explicit to preserve full-BPTT by default."""
        return RWKVState(
            spec=self.spec,
            layers=tuple(layer.clone(detach=detach) for layer in self.layers),
            schema_version=self.schema_version,
        )

    def detached(self) -> RWKVState:
        return self.clone(detach=True)

    def tensor_dict(self, *, detach: bool = True) -> dict[str, Any]:
        torch = _torch()

        def prepare(tensor):
            return tensor.detach().clone() if detach else tensor

        return {
            "time_mix_previous_x": torch.stack(
                [prepare(layer.time_mix_previous_x) for layer in self.layers]
            ),
            "wkv_matrix": torch.stack([prepare(layer.wkv_matrix) for layer in self.layers]),
            "channel_mix_previous_x": torch.stack(
                [prepare(layer.channel_mix_previous_x) for layer in self.layers]
            ),
        }

    def save(self, path: str | Path) -> None:
        torch = _torch()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": self.schema_version,
                "spec": asdict(self.spec),
                "tensors": self.tensor_dict(detach=True),
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: Any = None) -> RWKVState:
        torch = _torch()
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("RWKV state file must contain a mapping")
        if set(payload) != {"schema_version", "spec", "tensors"}:
            raise ValueError("RWKV state file keys do not match schema")
        if payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("RWKV state file has an unsupported schema version")
        spec_values = payload["spec"]
        expected_spec_keys = {
            "n_layer",
            "n_embd",
            "n_head",
            "head_size",
            "batch_size",
        }
        if not isinstance(spec_values, dict) or set(spec_values) != expected_spec_keys:
            raise ValueError("RWKV state spec keys do not match schema")
        spec = RWKVStateSpec(**spec_values)
        tensors = payload["tensors"]
        expected_keys = {
            "time_mix_previous_x",
            "wkv_matrix",
            "channel_mix_previous_x",
        }
        if not isinstance(tensors, dict) or set(tensors) != expected_keys:
            raise ValueError("RWKV state tensor keys do not match schema")
        layers = tuple(
            RWKVLayerState(
                time_mix_previous_x=tensors["time_mix_previous_x"][index],
                wkv_matrix=tensors["wkv_matrix"][index],
                channel_mix_previous_x=tensors["channel_mix_previous_x"][index],
            )
            for index in range(spec.n_layer)
        )
        return cls(spec=spec, layers=layers, schema_version=payload["schema_version"])
