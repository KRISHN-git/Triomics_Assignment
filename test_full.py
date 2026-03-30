from dotenv import load_dotenv
load_dotenv()

import json
import os
from utils.loader       import load_patient_notes
from pipeline.extractor import extract_from_note
from pipeline.merger    import merge_and_classify
from pipeline.validator import validate_conditions

PATIENT_ID = "patient_16"

print(f"=== Full pipeline test: {PATIENT_ID} ===\n")

notes = load_patient_notes("data/train", PATIENT_ID)
print(f"Loaded {len(notes)} notes")

all_extractions = {}
note_dates = {}
for note in notes:
    print(f"  Extracting {note['note_id']}...", end=" ", flush=True)
    extractions = extract_from_note(note)
    all_extractions[note["note_id"]] = extractions
    note_dates[note["note_id"]]      = note["encounter_date"]
    print(f"{len(extractions)} conditions")

print(f"\nTotal raw: {sum(len(v) for v in all_extractions.values())} condition mentions")

print("\nMerging and classifying...")
conditions = merge_and_classify(all_extractions, note_dates)
print(f"After merge: {len(conditions)} unique conditions")

validated = validate_conditions(conditions, PATIENT_ID)
print(f"After validation: {len(validated)} valid conditions")

print("\n=== FINAL CONDITIONS ===")
for c in validated:
    icon = {"active":"✓","resolved":"↩","suspected":"?"}
    print(f"  {icon.get(c['status'],'·')} [{c['category']}.{c['subcategory']}] {c['condition_name']}")
    print(f"    onset={c.get('onset')} | evidence in {len(c.get('evidence',[]))} notes")

os.makedirs("output", exist_ok=True)
with open(f"output/{PATIENT_ID}.json", "w") as f:
    json.dump({"patient_id": PATIENT_ID, "conditions": validated}, f, indent=2)
print(f"\nSaved to output/{PATIENT_ID}.json")
print(f"Compare against: data/train/labels/{PATIENT_ID}.json")