from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from src.training.checkpoint import PROVENANCE_KEYS, load_checkpoint, save_checkpoint
from src.training.episode_collator import PackedEpisodeCollator
from src.training.packing import CausalSequence
from src.training.parameter_groups import FP32MasterAdamW, OptimizerGroup, OptimizerPlan
from src.training.sft_trainer import PackedAgentSFTTrainer
from src.training.token_budget_sampler import TokenBudgetPackSampler


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(16, 8)
        self.dropout = torch.nn.Dropout(0.25)
        self.head = torch.nn.Linear(8, 16, bias=False)

    def _forward_features(self, input_ids, starts):
        return self.dropout(self.emb(input_ids))


def make_trainer():
    model = Model().to(dtype=torch.bfloat16)
    names, parameters = zip(*model.named_parameters())
    group = OptimizerGroup("base", tuple(names), tuple(parameters), 1.0, 0.01)
    optimizer = FP32MasterAdamW(OptimizerPlan((), (group,)), learning_rate=0.01)
    return PackedAgentSFTTrainer(model, optimizer, gradient_clip_norm=1.0)


def make_sampler(seed=19):
    return TokenBudgetPackSampler([10, 10], ["a", "b"], max_tokens=16, seed=seed)


def batch():
    sequence = CausalSequence("a", (1, 2, 3), (2, 3, 4), (0.0, 1.0, 1.0))
    return PackedEpisodeCollator(object(), max_pack_tokens=16).collate_tokenized([sequence])


def test_interrupted_fp32_master_training_matches_continuous_with_rng_and_cursor(tmp_path):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    trainer, sampler = make_trainer(), make_sampler()
    trainer.train_step(batch())
    provenance = {name: "a" * 64 for name in PROVENANCE_KEYS}
    path = tmp_path / "step1.pt"
    save_checkpoint(path, trainer, sampler, next_row=1, provenance=provenance)
    expected_random = (random.random(), float(np.random.rand()), torch.rand(5))
    expected_metrics = trainer.train_step(batch())
    expected_state = {name: p.detach().clone() for name, p in trainer.network.named_parameters()}
    restored, restored_sampler = make_trainer(), make_sampler()
    assert load_checkpoint(path, restored, restored_sampler, provenance=provenance) == 1
    assert restored_sampler.plan().rank_rows[1] == sampler.plan().rank_rows[1]
    actual_random = (random.random(), float(np.random.rand()), torch.rand(5))
    assert actual_random[:2] == expected_random[:2]
    assert torch.equal(actual_random[2], expected_random[2])
    assert restored.train_step(batch()) == expected_metrics
    assert restored.progress == trainer.progress
    for name, p in restored.network.named_parameters():
        assert torch.equal(p, expected_state[name])
    for expected, actual in zip(
        trainer.optimizer.state_dict()["master_weights"],
        restored.optimizer.state_dict()["master_weights"],
        strict=True,
    ):
        assert torch.equal(expected, actual)
    assert restored.optimizer.optimizer_state_dtypes == {torch.float32}
    with pytest.raises(FileExistsError):
        save_checkpoint(path, restored, sampler, next_row=1, provenance=provenance)


def test_checkpoint_rejects_changed_provenance_and_sampler(tmp_path):
    trainer, sampler = make_trainer(), make_sampler()
    provenance = {name: "a" * 64 for name in PROVENANCE_KEYS}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, trainer, sampler, next_row=0, provenance=provenance)
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(
            path, trainer, sampler, provenance={**provenance, "config_sha256": "b" * 64}
        )
    incompatible = TokenBudgetPackSampler([9, 11], ["a", "b"], max_tokens=16)
    with pytest.raises(ValueError, match="sampler identity"):
        load_checkpoint(path, trainer, incompatible, provenance=provenance)
