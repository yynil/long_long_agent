"""Separate cause-isolation diagnostics from the frozen generation acceptance run."""

import json
from pathlib import Path

import torch

from .generation_parity import PROMPTS, comparison
from .inference_math import diagnostic_reference_rows
from .runtime import load_runtime
from .rwkv7_stateful import state_passing, state_passing_inference, stateful_forward


@torch.inference_mode()
def diagnose(role: str, lm: Path, cuda: Path, build: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite diagnostic")
    model, tokenizer, provenance = load_runtime(role, lm, cuda, build)
    rows = []
    for length in (16, 32, 64, 128):
        ids = tokenizer.encode(PROMPTS[0] * 4)[:length]
        tokens = torch.tensor([ids], device="cuda")
        reference = model(tokens)
        with diagnostic_reference_rows(length):
            state, parts = None, []
            for token in tokens[0]:
                logits, state = stateful_forward(model, token.view(1, 1), state=state)
                parts.append(logits)
        # The alignment is deliberately the reference's shape: proves attribution,
        # not a shape-independent deployment solution or an acceptance rerun.
        rows.append({"matched_matrix_rows": length, **comparison(reference, torch.cat(parts, 1))})
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    values = [
        torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1
        for _ in range(6)
    ]
    initial = torch.randn(1, 2, 64, 64, device="cuda", generator=generator) * 0.1
    starts = torch.zeros(1, 32, device="cuda", dtype=torch.uint8)
    starts[:, 13] = 1
    whole, state = state_passing(*values, initial, starts)
    short_state, short_outputs = initial, []
    for index in range(32):
        part, short_state = state_passing_inference(
            *(x[:, index : index + 1] for x in values), short_state, starts[:, index : index + 1]
        )
        short_outputs.append(part)
    kernel = {
        "whole_vs_single_output_exact": bool(torch.equal(whole, torch.cat(short_outputs, 1))),
        "whole_vs_single_state_exact": bool(torch.equal(state, short_state)),
        "nonzero_initial_state": True,
        "internal_reset_position": 13,
    }
    result = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "provenance": provenance,
        "matched_shape_comparisons": rows,
        "kernel": kernel,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
