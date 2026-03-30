import os
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

def get_client():
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY")
    )

def get_model():
    return os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile")


MERGER_SYSTEM_PROMPT = """You are a senior medical coding specialist.
Classify medical conditions using the provided taxonomy.
Return ONLY a valid JSON array. No markdown. No explanation. No extra text.
Start with [ and end with ]."""

MERGER_USER_PROMPT = """Merge these per-note medical condition extractions for one patient.

TASKS:
1. MERGE duplicates — same condition across notes = ONE entry
2. CLASSIFY — assign category.subcategory from taxonomy
3. STATUS — use LATEST note's signals (active/resolved/suspected)
4. ONSET — earliest explicit date
5. EVIDENCE — all mentions from all notes

TAXONOMY:
cancer: primary_malignancy, metastasis, pre_malignant, benign, cancer_of_unknown_primary
cardiovascular: coronary, hypertensive, rhythm, vascular, structural, inflammatory_vascular
infectious: bacterial, viral, fungal, parasitic, spirochetal
metabolic_endocrine: diabetes, thyroid, genetic_metabolic, nutritional_deficiency, lipid, adrenal, pituitary
neurological: cerebrovascular, traumatic, seizure, functional, degenerative, neuromuscular
pulmonary: obstructive, acute_respiratory, structural, occupational, cystic
gastrointestinal: hepatic, biliary, upper_gi, lower_gi, inflammatory_bowel, functional_gi
renal: renal_failure, structural, glomerular, renovascular
hematological: cytopenia, coagulation, hemoglobinopathy
immunological: immunodeficiency, allergic, autoimmune, autoinflammatory, complement_deficiency
musculoskeletal: fracture, degenerative, crystal_arthropathy, connective_tissue_disorder
toxicological: poisoning, environmental_exposure
dental_oral: dental, temporomandibular

RULES:
- Heart failure → classify by cause (CAD→coronary, HTN→hypertensive, unknown→structural)
- Diabetic complications → metabolic_endocrine.diabetes (NOT renal/neuro)
- ANY low blood count → hematological.cytopenia regardless of cause
- Metastases → ONE entry PER site (brain≠liver≠bone≠lung)
- Esophageal varices → gastrointestinal.upper_gi
- Drug allergies → immunological.allergic
- Procedures (surgery, transplant) → SKIP, not conditions
- status active: in Diagnoses section, currently treated
- status resolved: "history of", "status post", only in Medical History
- status suspected: "suspected", "possible", "rule out"
- onset: use stated date first, else note encounter date, else null

Note encounter dates: {note_dates_json}

Per-note extractions: {extractions_json}
"""

def clean_json_response(raw: str) -> str:
    """Cleans LLM output to extract valid JSON."""
    raw = raw.strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    if not raw.startswith("[") and not raw.startswith("{"):
        bracket = raw.find("[")
        brace   = raw.find("{")
        if bracket == -1 and brace == -1:
            return raw
        elif bracket == -1:
            raw = raw[brace:]
        elif brace == -1:
            raw = raw[bracket:]
        else:
            raw = raw[min(bracket, brace):]

    def fix_newlines_in_strings(text):
        result = []
        in_string = False
        escape_next = False
        for char in text:
            if escape_next:
                result.append(char)
                escape_next = False
            elif char == '\\':
                result.append(char)
                escape_next = True
            elif char == '"':
                in_string = not in_string
                result.append(char)
            elif char in ('\n', '\r') and in_string:
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)

    raw = fix_newlines_in_strings(raw)

    raw = raw.strip()
    if not raw.endswith("]"):
        last_complete = raw.rfind("},")
        if last_complete == -1:
            last_complete = raw.rfind("}")
        if last_complete != -1:
            raw = raw[:last_complete + 1] + "\n]"

    return raw.strip()


def compress_extractions(all_extractions: dict) -> dict:
    """
    Compresses raw extractions to reduce token count.
    WHY: Groq has a 12,000 token/minute limit.
    Keeping only essential fields reduces prompt size significantly.
    """
    compressed = {}
    for note_id, conditions in all_extractions.items():
        compressed[note_id] = [
            {
                "condition_name": c.get("condition_name", ""),
                "section":        c.get("section", ""),
                "status_hint":    c.get("status_hint", ""),
                "onset_hint":     c.get("onset_hint", ""),
                "line_no":        c.get("line_no", 0),
                "span":           c.get("span", "")[:40]
            }
            for c in conditions
        ]
    return compressed


def merge_and_classify(all_extractions: dict, note_dates: dict) -> list:
    """
    Merges per-note extractions into a final unified condition list.
    Splits into batches if too many conditions to fit in one prompt.
    """
    compressed = compress_extractions(all_extractions)

    total = sum(len(v) for v in compressed.values())
    print(f"  Total compressed conditions: {total}")

    if total <= 80:
        return _merge_batch(compressed, note_dates)

    print(f"  Too many conditions ({total}), splitting into batches...")
    note_ids = list(compressed.keys())
    mid = len(note_ids) // 2

    batch1 = {k: compressed[k] for k in note_ids[:mid]}
    batch2 = {k: compressed[k] for k in note_ids[mid:]}

    # Dates for each batch
    dates1 = {k: note_dates[k] for k in note_ids[:mid] if k in note_dates}
    dates2 = {k: note_dates[k] for k in note_ids[mid:] if k in note_dates}

    print(f"  Batch 1: notes {note_ids[:mid]}")
    results1 = _merge_batch(batch1, dates1)

    print(f"  Batch 2: notes {note_ids[mid:]}")
    results2 = _merge_batch(batch2, dates2)

    print(f"  Final merge: combining {len(results1)} + {len(results2)} conditions...")
    combined = {"batch1": results1, "batch2": results2}
    return _merge_batch(combined, note_dates)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=15, max=120)
)
def _merge_batch(extractions: dict, note_dates: dict) -> list:
    """Single merger API call for one batch."""
    extractions_json = json.dumps(extractions, indent=1)
    note_dates_json  = json.dumps(note_dates)

    prompt = MERGER_USER_PROMPT.format(
        extractions_json=extractions_json,
        note_dates_json=note_dates_json
    )

    approx_tokens = len(prompt) // 4
    print(f"  Batch prompt ~{approx_tokens} tokens")

    response = get_client().chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": MERGER_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=4000
    )

    raw = response.choices[0].message.content
    cleaned = clean_json_response(raw)

    try:
        conditions = json.loads(cleaned)
        if not isinstance(conditions, list):
            return []
        return conditions
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Merger JSON parse error: {e}\n"
            f"Raw (first 500 chars): {raw[:500]}"
        )
