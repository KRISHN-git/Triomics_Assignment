from dotenv import load_dotenv
load_dotenv()  

import argparse    
import json
import logging
import os
import sys
import time

from pathlib import Path
from tqdm import tqdm
from utils.loader       import load_patient_notes
from pipeline.extractor import extract_from_note
from pipeline.merger    import merge_and_classify
from pipeline.validator import validate_conditions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def check_environment():
    required = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"]
    missing  = [v for v in required if not os.environ.get(v)]
    if missing:
        print("\nERROR: Missing environment variables:")
        for v in missing:
            print(f"  {v}")
        print("\nCreate a .env file with:")
        print("  OPENAI_BASE_URL=https://api.groq.com/openai/v1")
        print("  OPENAI_API_KEY=gsk_your-key-here")
        print("  OPENAI_MODEL=llama-3.3-70b-versatile")
        sys.exit(1)


def process_patient(data_dir, patient_id, output_dir, verbose=False):

    stats = {
        "patient_id": patient_id,
        "notes_processed": 0,
        "raw_extractions": 0,
        "final_conditions": 0,
        "error": None
    }

    try:
        notes = load_patient_notes(data_dir, patient_id)
        stats["notes_processed"] = len(notes)
        if verbose:
            logger.info(f"[{patient_id}] Loaded {len(notes)} notes")

        all_extractions = {}
        note_dates = {}
        for note in notes:
            if verbose:
                logger.info(f"[{patient_id}]   Extracting {note['note_id']}...")
            extractions = extract_from_note(note)
            all_extractions[note["note_id"]] = extractions
            note_dates[note["note_id"]]      = note["encounter_date"]
            stats["raw_extractions"] += len(extractions)
            if verbose:
                logger.info(f"[{patient_id}]   → {len(extractions)} raw conditions")

        if verbose:
            logger.info(f"[{patient_id}] Merging and classifying...")
        conditions = merge_and_classify(all_extractions, note_dates)

        validated = validate_conditions(conditions, patient_id)
        stats["final_conditions"] = len(validated)

        output   = {"patient_id": patient_id, "conditions": validated}
        out_path = Path(output_dir) / f"{patient_id}.json"
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"[{patient_id}]  {len(validated)} conditions → {out_path}")

    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"[{patient_id}] FAILED: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured medical conditions from clinical notes."
    )
    parser.add_argument("--data-dir",     required=True, help="Path to data directory")
    parser.add_argument("--patient-list", required=True, help="Path to patients JSON file")
    parser.add_argument("--output-dir",   required=True, help="Path to output directory")
    parser.add_argument("--verbose",    action="store_true", help="Detailed per-note logging")
    parser.add_argument("--skip-done",  action="store_true", help="Skip already-processed patients")

    args = parser.parse_args()
    check_environment()

    patient_list_path = Path(args.patient_list)
    if not patient_list_path.exists():
        print(f"ERROR: Patient list not found: {patient_list_path}")
        sys.exit(1)

    patient_ids = json.loads(patient_list_path.read_text(encoding="utf-8"))
    print(f"\nPatients  : {len(patient_ids)}")
    print(f"Data dir  : {args.data_dir}")
    print(f"Output dir: {args.output_dir}\n")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_stats = []
    for patient_id in tqdm(patient_ids, desc="Patients", unit="patient"):
        out_path = Path(args.output_dir) / f"{patient_id}.json"
        if args.skip_done and out_path.exists():
            logger.info(f"[{patient_id}] Skipping (already done)")
            continue
        stats = process_patient(args.data_dir, patient_id, args.output_dir, args.verbose)
        all_stats.append(stats)
        time.sleep(10)

    successful = [s for s in all_stats if s["error"] is None]
    failed     = [s for s in all_stats if s["error"] is not None]
    print(f"\n{'='*50}")
    print(f"Completed : {len(successful)} / {len(all_stats)}")
    print(f"Failed    : {len(failed)}")
    if successful:
        avg = sum(s["final_conditions"] for s in successful) / len(successful)
        print(f"Avg conditions/patient: {avg:.1f}")
    if failed:
        print("\nFailed patients:")
        for s in failed:
            print(f"  {s['patient_id']}: {s['error']}")
    print(f"\nOutputs written to: {args.output_dir}/")


if __name__ == "__main__":
    main()