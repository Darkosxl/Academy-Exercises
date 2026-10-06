"""Collect results/*.json into a leaderboard ranked by test macro-F1."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
res = [json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))]
res.sort(key=lambda r: (-r["test_f1"], r["params_m"]))

lines = ["| Model | Params | Val F1 | Test F1 | Test acc | Best ep | Train time | Latency MPS | Latency CPU | Errors (of 407) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in res:
    errs = sum(v for i, row in enumerate(r["confusion"]) for j, v in enumerate(row) if i != j)
    lines.append(
        f"| {r['model']} | {r['params_m']:.1f}M | {r['val_f1']:.4f} | **{r['test_f1']:.4f}** | {r['test_acc']:.4f} "
        f"| {r['best_epoch']} | {r['train_time_s'] / 60:.1f} min | {r['latency_ms_mps'] or float('nan'):.1f} ms "
        f"| {r['latency_ms_cpu']:.1f} ms | {errs} |"
    )
table = "\n".join(lines)
print(table)
(ROOT / "reports" / "leaderboard.md").write_text(table + "\n")
