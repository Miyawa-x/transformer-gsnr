import json
from pathlib import Path

import pytest

from transformer_gsnr.train import run, split_indices, validate_config


def test_split_is_disjoint_balanced_reproducible():
    targets = [label for label in range(100) for _ in range(500)]
    split = split_indices(targets, 42)
    assert split == split_indices(targets, 42)
    assert sorted(map(len, split.values())) == [5000, 5000, 40000]
    values = [i for v in split.values() for i in v]
    assert len(set(values)) == 50000
    for indices in split.values():
        counts = {label: sum(targets[i] == label for i in indices) for label in range(100)}
        assert len(set(counts.values())) == 1


def test_all_configs_validate():
    configs = Path(__file__).resolve().parents[1] / "configs"
    for path in configs.glob("*.json"):
        validate_config(json.loads(path.read_text()))


def test_training_requires_clean_decay_control():
    path = Path(__file__).resolve().parents[1] / "configs/ce_adaptive.json"
    config = json.loads(path.read_text())
    config["weight_decay"] = .05
    with pytest.raises(ValueError, match="zero decay"):
        validate_config(config)


def test_dataset_training_cannot_run_on_local_cpu(tmp_path):
    from argparse import Namespace
    with pytest.raises(RuntimeError, match="CUDA server"):
        run(Namespace(device="cpu", output=str(tmp_path / "forbidden")))
    assert not (tmp_path / "forbidden").exists()
