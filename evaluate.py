from dotenv import load_dotenv
load_dotenv()

import json
import argparse
from pathlib import Path


def normalize(name: str) -> str:
    """Lowercase and strip for fuzzy name matching."""
    return name.lower().strip()


def score_patient(pred_path, gold_path, verbose=False):
    pred = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))

    pred_conds = pred.get("conditions", [])
    gold_conds = gold.get("conditions", [])

    pred_names = {normalize(c["condition_name"]) for c in pred_conds}
    gold_names = {normalize(c["condition_name"]) for c in gold_conds}

    tp = pred_names & gold_names
    fp = pred_names - gold_names
    fn = gold_names - pred_names

    precision = len(tp) / len(pred_names) if pred_names else 0
    recall    = len(tp) / len(gold_names) if gold_names else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

    status_correct = 0
    for pc in pred_conds:
        pname = normalize(pc["condition_name"])
        if pname in gold_names:
            for gc in gold_conds:
                if normalize(gc["condition_name"]) == pname:
                    if pc.get("status") == gc.get("status"):
                        status_correct += 1
                    break
    status_acc = status_correct / len(tp) if tp else 0

    if verbose:
        pid = pred.get("patient_id", Path(pred_path).stem)
        print(f"\n{'─'*50}")
        print(f"Patient: {pid}")
        print(f"  Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}")
        print(f"  Status accuracy: {status_acc:.2f}")
        if fn:
            print(f"  MISSED ({len(fn)}):")
            for n in sorted(fn):
                print(f"    ✗ {n}")
        if fp:
            print(f"  FALSE POSITIVES ({len(fp)}):")
            for n in sorted(fp):
                print(f"    + {n}")

    return {
        "patient_id":  pred.get("patient_id"),
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "status_acc":  status_acc,
        "tp": len(tp), "fp": len(fp), "fn": len(fn)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline output against ground truth labels."
    )
    parser.add_argument("--pred-dir", required=True, help="Your output folder")
    parser.add_argument("--gold-dir", required=True, help="Ground truth labels folder")
    parser.add_argument("--verbose",  action="store_true", help="Show per-patient details")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gold_dir = Path(args.gold_dir)

    scores = []
    for gold_path in sorted(gold_dir.glob("*.json")):
        pred_path = pred_dir / gold_path.name
        if not pred_path.exists():
            print(f"WARNING: No prediction for {gold_path.name}")
            continue
        score = score_patient(pred_path, gold_path, verbose=args.verbose)
        scores.append(score)

    if not scores:
        print("No scores computed. Check your paths.")
        return

    print(f"\n{'='*50}")
    print("OVERALL SCORES")
    print(f"{'='*50}")
    print(f"Patients evaluated : {len(scores)}")
    print(f"Avg Precision      : {sum(s['precision']  for s in scores)/len(scores):.3f}")
    print(f"Avg Recall         : {sum(s['recall']     for s in scores)/len(scores):.3f}")
    print(f"Avg F1             : {sum(s['f1']         for s in scores)/len(scores):.3f}")
    print(f"Avg Status Acc     : {sum(s['status_acc'] for s in scores)/len(scores):.3f}")
    print()
    print("Per-patient F1:")
    for s in scores:
        bar = "█" * int(s["f1"] * 20)
        print(f"  {s['patient_id']:15s}  F1={s['f1']:.3f}  {bar}")


if __name__ == "__main__":
    main()