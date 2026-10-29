import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from transformer_gsnr.control import MomentumMap, calibrate, diagnostic_rates, make_optimizer, update_momenta
from transformer_gsnr.model import SmallViT, ViTConfig
from transformer_gsnr.task import DistillationTask, gradients, parameter_layout
from transformer_gsnr.vendor.losses import dkd_component_losses
from transformer_gsnr.vendor.measurement import GradientWindow, rng_state, state_hash


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def small(width=12, depth=1, dropout=0.):
    return SmallViT(ViTConfig(image_size=8, patch_size=4, classes=6, width=width,
                              depth=depth, heads=3, mlp_ratio=2, dropout=dropout))


def batches(n=4):
    generator = torch.Generator().manual_seed(91)
    return [(torch.randn(3, 3, 8, 8, generator=generator),
             torch.randint(6, (3,), generator=generator)) for _ in range(n)]


def test_exact_gsnr_with_bias_correction():
    p = torch.nn.Parameter(torch.zeros(2))
    window = GradientWindow([("layer.weight", p)])
    for values in ([1., 3.], [3., 1.]):
        window.observe([torch.tensor(values)])
    row = window.rows()[0]
    assert row["noise_trace"] == pytest.approx(4.)
    assert row["signal_unclipped"] == pytest.approx(6.)
    assert row["gsnr"] == pytest.approx(1.5)


def test_zero_and_inactive_gradients_are_zero_observations():
    window = GradientWindow([("layer.w", torch.zeros(1))])
    window.observe([torch.ones(1)])
    window.observe([None])
    row = window.rows()[0]
    assert row["observations"] == 2
    assert row["noise_trace"] == pytest.approx(.5)
    assert row["signal_unclipped"] == pytest.approx(0.)


@pytest.mark.parametrize("scale", [.1, 10.])
def test_scale_invariance_of_gsnr(scale):
    a, b = [GradientWindow([("layer.w", torch.zeros(2))]) for _ in range(2)]
    for values in ([1., 3.], [3., 1.], [2., 4.]):
        g = torch.tensor(values, dtype=torch.float64)
        a.observe([g])
        b.observe([g * scale])
    ar, br = a.rows()[0], b.rows()[0]
    assert br["gsnr"] == pytest.approx(ar["gsnr"], rel=1e-8)
    assert br["noise_trace"] == pytest.approx(ar["noise_trace"] * scale**2)


def test_monotone_map_and_huge_finite_value():
    mapping = MomentumMap()
    values = [mapping(s) for s in [0., .01, 1., 100., 1e308]]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert values[0] == mapping.minimum and values[-1] <= mapping.maximum


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_signal_rejected(value):
    with pytest.raises(ValueError):
        MomentumMap()(value)


@pytest.mark.parametrize("mode", ["ce", "kd", "feature"])
def test_parameter_layout_and_finite_transformer_backward(mode):
    task = DistillationTask(small(), mode=mode, teacher_width=18)
    teacher = small(width=18).eval()
    named, layers = parameter_layout(task)
    assert len(named) == len(layers) == len(list(task.parameters()))
    assert {"embedding", "blocks.0.attn", "blocks.0.mlp", "blocks.0.norm1", "blocks.0.norm2", "final_norm", "head"} <= set(layers)
    images, targets = batches()[0]
    gs = gradients(task(images, targets, teacher), [p for _, p in named])
    for stream in task.streams:
        assert any(g is not None and g.norm() > 0 for g in gs[stream])
        assert all(g is None or torch.isfinite(g).all() for g in gs[stream])
    assert all(p.grad is None for p in teacher.parameters())
    if mode == "feature":
        assert all(g is None for g, layer in zip(gs["feature"], layers) if layer == "head")
        assert any(g is not None for g, layer in zip(gs["feature"], layers) if layer == "feature_adapter")


def test_original_dkd_losses_are_preserved():
    student, teacher = small(), small().eval()
    task = DistillationTask(student, mode="kd", temperature=4.)
    images, targets = batches()[0]
    losses = task(images, targets, teacher)
    zs, _ = student(images)
    with torch.no_grad():
        zt, _ = teacher(images)
    tcg, ncg = dkd_component_losses(zs, zt, targets, 4.)
    assert torch.equal(losses["tcg"], tcg)
    assert torch.equal(losses["ncg"], ncg)


def test_probe_restores_rng_modes_parameters_gradients_and_dropout():
    task = DistillationTask(small(dropout=.4), mode="feature", teacher_width=18)
    teacher = small(width=18, dropout=.3).train()
    task.student.blocks[0].norm1.eval()  # Restore mixed submodule modes too.
    next(task.parameters()).grad = torch.ones_like(next(task.parameters()))
    original_rng = state_hash(rng_state())
    result = calibrate(task, batches(), teacher)
    assert result["isolation"]["unchanged"]
    assert state_hash(rng_state()) == original_rng
    assert teacher.training and task.training
    assert not task.student.blocks[0].norm1.training
    second = calibrate(task, batches(), teacher)
    assert second["rows"] == result["rows"]
    for row in result["rows"]:
        assert row["observations"] == 4
        if row["layer"] == "head" and row["stream"] == "feature":
            assert row["active_observations"] == 0


def test_probe_error_also_restores_state():
    task = DistillationTask(small())
    before = state_hash([task.state_dict(), rng_state()])
    data = batches()
    data[-1] = (data[-1][0][:1], data[-1][1][:1])
    with pytest.raises(ValueError, match="equal"):
        calibrate(task, data)
    assert state_hash([task.state_dict(), rng_state()]) == before


def test_equal_momentum_multistream_matches_pytorch_sgd():
    a = DistillationTask(small(), mode="kd")
    b = copy.deepcopy(a)
    teacher = small().eval()
    ours = make_optimizer(a, lr=.01, momentum=.8, weight_decay=.001)
    reference = torch.optim.SGD(b.parameters(), lr=.01, momentum=.8, weight_decay=.001)
    for images, targets in batches(3):
        ours.step(gradients(a(images, targets, teacher), ours.ordered_parameters()))
        reference.zero_grad()
        sum(b(images, targets, teacher).values()).backward()
        reference.step()
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_coefficient_change_keeps_buffer_history_and_checkpoint_roundtrip():
    task = DistillationTask(small())
    optimizer = make_optimizer(task, lr=.01)
    images, targets = batches()[0]
    optimizer.step(gradients(task(images, targets), optimizer.ordered_parameters()))
    previous = {id(p): optimizer.state[p]["momentum_buffer_tog"].clone() for p in task.parameters()}
    result = calibrate(task, batches())
    updates = update_momenta(optimizer, result, MomentumMap())
    assert updates
    for p in task.parameters():
        assert torch.equal(previous[id(p)], optimizer.state[p]["momentum_buffer_tog"])
    restored = copy.deepcopy(task)
    restored_optimizer = make_optimizer(restored, lr=.01)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for model, opt in [(task, optimizer), (restored, restored_optimizer)]:
        opt.step(gradients(model(images, targets), opt.ordered_parameters()))
    for left, right in zip(task.parameters(), restored.parameters()):
        assert torch.equal(left, right)


def test_rates_do_not_invent_log_decay_at_zero():
    old = {"rows": [{"layer": "head", "stream": "ncg", "gsnr": 2.}]}
    new = {"rows": [{"layer": "head", "stream": "ncg", "gsnr": 0.}]}
    rate = diagnostic_rates(new, old, 10)[0]
    assert rate["absolute_drop_per_epoch"] == .2
    assert rate["log_decay_per_epoch"] is None


def test_vendored_sources_unchanged():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "UPSTREAMS.json").read_text())
    for item in manifest["files"]:
        assert hashlib.sha256((root / item["destination"]).read_bytes()).hexdigest() == item["sha256"]
