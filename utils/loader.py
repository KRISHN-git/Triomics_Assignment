from pathlib import Path
import re

def extract_encounter_date(lines: list[str]) -> str:
   
    for line in lines[:10]:
        match = re.search(r'(\d{2}/\d{2}/\d{4})', line)
        if match:
            from datetime import datetime
            try:
                dt = datetime.strptime(match.group(1), "%m/%d/%Y")
                return dt.strftime("%B %Y")
            except ValueError:
                return match.group(1)
    return "unknown"  


def load_patient_notes(data_dir: str, patient_id: str) -> list[dict]:
    
    patient_dir = Path(data_dir) / patient_id
    
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")
    
    note_files = sorted(patient_dir.glob("text_*.md"))
    
    if not note_files:
        raise FileNotFoundError(f"No text_*.md files found in {patient_dir}")
    
    notes = []
    for path in note_files:
        raw_text = path.read_text(encoding="utf-8")
        lines = raw_text.splitlines()
        
        numbered_lines = [f"{i+1}: {line}" for i, line in enumerate(lines)]
        numbered_text = "\n".join(numbered_lines)
        
        notes.append({
            "note_id": path.stem,                         
            "encounter_date": extract_encounter_date(lines),
            "lines": lines,                  
            "numbered_text": numbered_text
        })
    
    return notes