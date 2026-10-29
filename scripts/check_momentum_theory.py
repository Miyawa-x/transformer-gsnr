"""Frozen gradient sequence calculation, not neural-network training."""
import json
from pathlib import Path
import numpy as np


def main():
    rows = []
    # Same gradient multiset: [1, +1] and [1, -1], only order differs.
    for mu in (.5, .9, .95):
        for kind, signs in [("alternating", np.tile([1., -1.], 64)),
                            ("blocks", np.repeat([1., -1.], 64))]:
            gradients = np.column_stack((np.ones(128), signs))
            v = np.zeros(2)
            buffers = []
            for period in range(100):
                for g in gradients:
                    v = mu * v + g  # Original unnormalized Heavy-Ball.
                    if period == 99:
                        buffers.append(v.copy())
            buffers = np.asarray(buffers)
            mean = buffers.mean(0)
            variance = ((buffers - mean)**2).sum(1).mean()
            rows.append({"momentum": mu, "order": kind, "input_population_gsnr": 1.,
                "buffer_population_gsnr": float(mean @ mean / variance),
                "buffer_noise": float(variance), "mean_gain": float(mean[0]),
                "iid_formula_would_predict": (1 + mu) / (1 - mu)})
    for mu in (.5, .9, .95):
        pair = [r for r in rows if r["momentum"] == mu]
        assert pair[0]["buffer_population_gsnr"] > pair[1]["buffer_population_gsnr"]
        assert all(np.isclose(r["mean_gain"], 1 / (1 - mu)) for r in pair)
    result = {"scope": "synthetic periodic gradient replay; no student training", "rows": rows,
              "finding": "Equal instantaneous GSNR does not determine momentum-buffer GSNR; temporal correlation matters."}
    destination = Path(__file__).resolve().parents[1] / "reports/momentum-theory-check.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
