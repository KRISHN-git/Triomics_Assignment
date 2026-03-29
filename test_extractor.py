from dotenv import load_dotenv
load_dotenv()

import os
from utils.loader import load_patient_notes
from pipeline.extractor import extract_from_note

PATIENT_ID = "patient_06"

print(f"Model: {os.environ.get('OPENAI_MODEL')}")
print(f"Base URL: {os.environ.get('OPENAI_BASE_URL')}")
print()

notes = load_patient_notes("data/train", PATIENT_ID)
print(f"Loaded {len(notes)} notes\n")

total = 0
for note in notes:
    print(f"Extracting {note['note_id']} ({note['encounter_date']})...", end=" ", flush=True)
    conditions = extract_from_note(note)
    print(f"{len(conditions)} conditions found")
    for c in conditions:
        print(f"  Line {c['line_no']:3d} [{c['section']}]: {c['condition_name']}")
    total += len(conditions)
    print()

print(f"TOTAL raw extractions across all notes: {total}")
print("(These will be deduplicated in the merger step)")