"""Run a paired Tetris trace replay and write its measurements as JSON."""

import argparse
import json
import math
from pathlib import Path

from src.pipeline import HyperSimulator


BETAS = (0.1, 0.3, 0.5, 0.7, 0.9)
ABLATIONS = {
    "none": (), "T1": (1,), "T2": (2,), "T3": (3,),
    "minus_T1": (2, 3), "minus_T2": (1, 3), "minus_T3": (1, 2),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--analytical", action="store_true", help="Use the analytical PIM model for a quick check.")
    parser.add_argument("--ablation", action="store_true", help="Include standalone and leave-one-out T1–T3 runs.")
    parser.add_argument("--steps", action="store_true", help="Include per-boundary timing and placement.")
    parser.add_argument("--orin-ms", type=float, help="Faster measured Orin stack latency for the same workload.")
    args = parser.parse_args()

    for path in (args.trace, args.model):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    if args.output.resolve() in (args.trace.resolve(), args.model.resolve()):
        parser.error("output must be separate from the input files")
    if args.orin_ms is not None and (not math.isfinite(args.orin_ms) or args.orin_ms <= 0):
        parser.error("--orin-ms must be a positive finite latency in milliseconds")

    def run(**settings):
        result = HyperSimulator(fast_pim_model=args.analytical, **settings).run(
            str(args.trace), str(args.model)
        )
        if not args.steps:
            del result["step_breakdown"]
        return result

    def tetris(techniques):
        return run(policy="ours", tech1=1 in techniques, tech2=2 in techniques, tech3=3 in techniques)

    def latency(result):
        return result["metrics"]["total_latency_ms"]

    runs = {"attacc": run(policy="baseline_static")}
    for name, boundary in (("orches_a", "branch"), ("orches_ab", "action")):
        candidates = {
            beta: run(policy="ours", orches=True, orches_boundary=boundary, beta_threshold=beta)
            for beta in BETAS
        }
        best = min(candidates, key=lambda beta: latency(candidates[beta]))
        runs[name] = candidates[best]
        runs[name]["selected_beta"] = best
        runs[name]["beta_scan_latency_ms"] = {beta: latency(row) for beta, row in candidates.items()}

    runs["tetris"] = tetris((1, 2, 3))
    if args.ablation:
        runs.update({name: tetris(techniques) for name, techniques in ABLATIONS.items()})

    if args.orin_ms is not None:
        for run_result in runs.values():
            run_result["speedup_vs_measured_orin"] = args.orin_ms / latency(run_result)

    report = {
        "trace": str(args.trace),
        "model_config": str(args.model),
        "pim_backend": "analytical" if args.analytical else "ramulator2",
        "orin_reference_ms": args.orin_ms,
        "speedup_over_orches_ab": latency(runs["orches_ab"]) / latency(runs["tetris"]),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
