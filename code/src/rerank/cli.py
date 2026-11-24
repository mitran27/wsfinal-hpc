"""Command line for the whole pipeline. Run `rerank -h` for the stage list.

Typical order:
    rerank index                      build the BM25 index
    rerank splits                     train / val / calib query lists
    rerank retrieve all               top-100 BM25 candidates per split
    rerank groups                     sampled training groups
    rerank encode train val calib dev dl19 dl20
    rerank train candidate            (and pointwise, ffn_control)
    rerank finetune candidate         LoRA backbone + head, end to end
    rerank predict <ckpt> dev dl19 dl20
    rerank calibrate <ckpt> --fit calib --apply dev dl19 dl20
    rerank evaluate dev runA runB --baseline runA
    rerank baseline bge|minilm dev dl19 dl20
    rerank stability <ckpt> dl19
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rerank.config import load_config
from rerank.data.splits import ALL_SPLITS

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pilot.toml"


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _splits(names: list[str]) -> list[str]:
    return list(ALL_SPLITS) if names == ["all"] else names


def cmd_index(cfg, args):
    from rerank.data.bm25 import build_index
    _print(build_index(cfg.data.collection, cfg.index_dir))


def cmd_splits(cfg, args):
    from rerank.data.splits import build_splits
    _print(build_splits(cfg))


def cmd_retrieve(cfg, args):
    from rerank.data.bm25 import BM25, retrieve_split
    from rerank.data.splits import split_spec
    index = BM25(cfg.index_dir, cfg.retrieval.k1, cfg.retrieval.b)
    for name in _splits(args.splits):
        _print(retrieve_split(cfg, split_spec(cfg, name), index))


def cmd_groups(cfg, args):
    from rerank.data.groups import build_training_groups
    _print(build_training_groups(cfg))


def cmd_encode(cfg, args):
    from rerank.features import encode_split
    from rerank.models.backbone import PairEncoder, load_backbone, pick_device, pick_dtype
    bb = cfg.backbone
    device = pick_device()
    encoder = PairEncoder(load_backbone(bb), bb.max_length, bb.max_query_tokens, device, pick_dtype(bb.dtype, device))
    for name in _splits(args.splits):
        print(encode_split(cfg, name, encoder, args.limit))


def cmd_train(cfg, args):
    from rerank.train.trainer import train
    if args.freeze_projection:
        cfg.train.freeze_projection = True
    print(train(cfg, args.variant, args.objective, args.seed, args.init, steps=args.steps))


def cmd_finetune(cfg, args):
    from rerank.train.finetune import finetune
    out_dir = cfg.models_dir / args.tag if args.tag else None
    print(finetune(cfg, args.variant, args.seed, args.steps, out_dir=out_dir, max_train_queries=args.max_train_queries))


def cmd_predict(cfg, args):
    from rerank.train.calibrate import PlattCalibrator
    from rerank.train.trainer import predict
    cal = PlattCalibrator.load(args.calibrator) if args.calibrator else None
    for name in args.splits:
        print(*predict(cfg, args.ckpt, name, cal, args.depth))


def cmd_calibrate(cfg, args):
    import numpy as np

    from rerank.data.splits import split_spec
    from rerank.eval import metrics
    from rerank.train.calibrate import PlattCalibrator, read_scores
    from rerank.train.trainer import predict

    tag = args.ckpt.parent.name
    _, fit_path = predict(cfg, args.ckpt, args.fit)
    z, y = read_scores(fit_path, split_spec(cfg, args.fit).threshold)
    cal = PlattCalibrator.fit(z, y)
    cal_path = cfg.models_dir / tag / f"calibrator.{args.fit}.json"
    cal.save(cal_path)
    report = {"calibrator": vars(cal), "fit_split": args.fit}
    for name in args.apply:
        _, path = predict(cfg, args.ckpt, name, cal)
        z, y = read_scores(path, split_spec(cfg, name).threshold)
        raw, calibrated = 1 / (1 + np.exp(-z)), cal.apply(z)
        report[name] = {
            "judged_pairs": int(len(y)), "judged_positive": int(y.sum()),
            "brier_raw": metrics.brier(raw, y), "brier_calibrated": metrics.brier(calibrated, y),
            "ece_raw": metrics.expected_calibration_error(raw, y),
            "ece_calibrated": metrics.expected_calibration_error(calibrated, y),
            "reliability_raw": metrics.reliability(raw, y),
            "reliability_calibrated": metrics.reliability(calibrated, y),
        }
    out = cfg.results_dir / f"calibration.{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    _print({k: v for k, v in report.items() if k == "calibrator"} | {
        n: {m: report[n][m] for m in ("judged_pairs", "brier_raw", "brier_calibrated")} for n in args.apply})


def cmd_evaluate(cfg, args):
    from rerank.data import io
    from rerank.data.splits import split_spec
    from rerank.eval import metrics

    spec = split_spec(cfg, args.split)
    qrels = spec.qrels()
    per_run = {}
    for path in args.runs:
        per_run[path.name] = metrics.per_query(io.read_run(path), qrels, spec.threshold)
    table = {name: metrics.aggregate(scores) for name, scores in per_run.items()}
    cands = cfg.candidates_dir / f"{args.split}.tsv"
    if cands.exists():
        table["oracle_ceiling"] = metrics.evaluate_run(metrics.ceiling_run(io.read_candidates(cands), qrels), qrels, spec.threshold)
    result = {"split": args.split, "metrics": table}
    if args.baseline:
        base = per_run[args.baseline.name]
        result["vs_" + args.baseline.name] = {
            name: {m: metrics.paired_bootstrap({q: s[m] for q, s in base.items()},
                                               {q: s[m] for q, s in scores.items()})
                   for m in ("mrr@10", "ndcg@10")}
            for name, scores in per_run.items() if name != args.baseline.name}
    out = cfg.results_dir / f"metrics.{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    _print(result)


def cmd_baseline(cfg, args):
    from rerank.systems import BiEncoder, CrossEncoder, rerank_split
    system = BiEncoder() if args.model == "bge" else CrossEncoder()
    for name in args.splits:
        print(rerank_split(cfg, system, name, args.depth, limit=args.limit))


def cmd_live(cfg, args):
    from rerank.systems import TrainedReranker, rerank_split
    system = TrainedReranker(cfg, args.ckpt)
    for name in args.splits:
        print(rerank_split(cfg, system, name, args.depth, tag=f"{system.name}-live", limit=args.limit))


def cmd_stability(cfg, args):
    from rerank.data.splits import split_spec
    from rerank.eval.stability import depth_test, shuffle_test
    from rerank.features import FeatureStore
    from rerank.train.trainer import load_checkpoint
    model, _ = load_checkpoint(args.ckpt)
    device = next(model.parameters()).device
    spec = split_spec(cfg, args.split)
    store = FeatureStore(cfg.feature_dir / args.split)
    _print({"shuffle": shuffle_test(model, store, device),
            "depth": depth_test(model, store, spec.qrels(), spec.threshold, device)})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="rerank", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override one config value, e.g. --set model.set_token=true (repeatable)")
    sub = p.add_subparsers(dest="cmd", required=True)
    splits_arg = dict(nargs="+", help=f"splits from {ALL_SPLITS}, or 'all'")

    sub.add_parser("index").set_defaults(fn=cmd_index)
    sub.add_parser("splits").set_defaults(fn=cmd_splits)
    s = sub.add_parser("retrieve"); s.add_argument("splits", **splits_arg); s.set_defaults(fn=cmd_retrieve)
    sub.add_parser("groups").set_defaults(fn=cmd_groups)
    s = sub.add_parser("encode"); s.add_argument("splits", **splits_arg)
    s.add_argument("--limit", type=int, help="encode only the first N queries (smoke tests)")
    s.set_defaults(fn=cmd_encode)

    s = sub.add_parser("train")
    s.add_argument("variant", choices=("pointwise", "candidate", "ffn_control"))
    s.add_argument("--objective", choices=("bce", "ranknet", "rl"), default="bce")
    s.add_argument("--seed", type=int)
    s.add_argument("--steps", type=int)
    s.add_argument("--init", type=Path, help="continue from this checkpoint")
    s.add_argument("--freeze-projection", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("finetune", help="LoRA on the backbone + head, trained end to end")
    s.add_argument("variant", choices=("pointwise", "candidate", "ffn_control"))
    s.add_argument("--seed", type=int); s.add_argument("--steps", type=int)
    s.add_argument("--max-train-queries", type=int)
    s.add_argument("--tag", help="output folder name under work/models/ (default ft-<variant>-s<seed>)")
    s.set_defaults(fn=cmd_finetune)

    s = sub.add_parser("predict"); s.add_argument("ckpt", type=Path); s.add_argument("splits", nargs="+")
    s.add_argument("--calibrator", type=Path); s.add_argument("--depth", type=int)
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("calibrate"); s.add_argument("ckpt", type=Path)
    s.add_argument("--fit", required=True, help="split with judged pairs of both classes")
    s.add_argument("--apply", nargs="+", default=[])
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("evaluate"); s.add_argument("split"); s.add_argument("runs", nargs="+", type=Path)
    s.add_argument("--baseline", type=Path, help="run to compare the others against (paired bootstrap)")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("baseline"); s.add_argument("model", choices=("bge", "minilm"))
    s.add_argument("splits", nargs="+"); s.add_argument("--depth", type=int, default=100)
    s.add_argument("--limit", type=int); s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser("live", help="end-to-end latency of a trained model, no feature cache")
    s.add_argument("ckpt", type=Path); s.add_argument("splits", nargs="+")
    s.add_argument("--depth", type=int, default=100); s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("stability"); s.add_argument("ckpt", type=Path); s.add_argument("split")
    s.set_defaults(fn=cmd_stability)

    args = p.parse_args(argv)
    args.fn(load_config(args.config, args.overrides), args)


if __name__ == "__main__":
    main()
