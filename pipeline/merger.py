import os
import json
import time
import re
from pathlib import Path
from openai import OpenAI, RateLimitError


def get_client():
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY")
    )

def get_model():
    return os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile")


def load_taxonomy_str() -> str:
    
    taxonomy_path = Path(__file__).parent.parent / "taxonomy.json"

    if taxonomy_path.exists():
        data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        lines = []
        for cat, info in data["condition_categories"].items():
            subcats = ", ".join(info["subcategories"].keys())
            lines.append(f"{cat}: {subcats}")
        return "\n".join(lines)

    print("WARNING: taxonomy.json not found, using hardcoded fallback")
    return (
        "cancer: primary_malignancy, metastasis, pre_malignant, benign, cancer_of_unknown_primary\n"
        "cardiovascular: coronary, hypertensive, rhythm, vascular, structural, inflammatory_vascular\n"
        "infectious: bacterial, viral, fungal, parasitic, spirochetal\n"
        "metabolic_endocrine: diabetes, thyroid, genetic_metabolic, nutritional_deficiency, lipid, adrenal, pituitary\n"
        "neurological: cerebrovascular, traumatic, seizure, functional, degenerative, neuromuscular\n"
        "pulmonary: obstructive, acute_respiratory, structural, occupational, cystic\n"
        "gastrointestinal: hepatic, biliary, upper_gi, lower_gi, inflammatory_bowel, functional_gi\n"
        "renal: renal_failure, structural, glomerular, renovascular\n"
        "hematological: cytopenia, coagulation, hemoglobinopathy\n"
        "immunological: immunodeficiency, allergic, autoimmune, autoinflammatory, complement_deficiency\n"
        "musculoskeletal: fracture, degenerative, crystal_arthropathy, connective_tissue_disorder\n"
        "toxicological: poisoning, environmental_exposure\n"
        "dental_oral: dental, temporomandibular"
    )


def load_disambiguation_rules() -> str:
    taxonomy_path = Path(__file__).parent.parent / "taxonomy.json"
    if taxonomy_path.exists():
        data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        rules = data.get("disambiguation_rules", [])
        return "\n".join(f"- {r['rule']}: {r['explanation']}" for r in rules)
    return ""


TAXONOMY_STR = load_taxonomy_str()
DISAMBIGUATION_STR = load_disambiguation_rules()


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

TAXONOMY (use EXACTLY these keys):
{taxonomy}

CRITICAL SUBCATEGORY RULES:
- pulmonary.structural → pneumothorax, pleural effusion, interstitial lung disease, pulmonary fibrosis
- pulmonary.acute_respiratory → ARDS, acute respiratory FAILURE only (NOT pneumothorax)
- neurological.traumatic → traumatic brain injury, skull fractures, spinal cord injury
- neurological.cerebrovascular → stroke, subarachnoid hemorrhage, TIA, intracranial hemorrhage
- cardiovascular.vascular → arterial dissection, aneurysm, DVT, peripheral vascular disease
- cardiovascular.inflammatory_vascular → vasculitis, arteritis ONLY
- musculoskeletal.fracture → bone fractures only (NOT cancer bone destruction → cancer.metastasis)
- hematological.cytopenia → ALL low blood counts regardless of cause

DISAMBIGUATION RULES (from official taxonomy):
{disambiguation}

ADDITIONAL RULES:
- ANY low blood count → ALWAYS hematological.cytopenia regardless of cause
- Metastases → ONE entry PER site (brain≠liver≠bone≠lung)
- Esophageal varices → gastrointestinal.upper_gi
- Drug/contrast allergies → immunological.allergic
- Procedures (surgery, biopsy, transplant) → SKIP, not conditions
- Nicotine/alcohol abuse → toxicological.environmental_exposure (NOT poisoning)
- Radiodermatitis → toxicological.environmental_exposure (NOT poisoning)

STATUS RULES (use LATEST note):
- active: in Diagnoses/Other Diagnoses section, currently treated, chronic, ongoing
- resolved: "history of", "status post", only in Medical History, "in remission"
- suspected: "suspected", "possible", "rule out", "likely", "suggestive of"

ONSET DATE RULES:
- Priority 1: explicitly stated date for condition ("diagnosed 05/2014" → "May 2014")
- Priority 2: use encounter date of EARLIEST note where condition appears (from Note encounter dates below)
- Priority 3: convert relative dates ("since mid-December" in Jan 2017 → "December 2016")
- Priority 4: null ONLY if truly no date can be determined
- NEVER return null if an encounter date is available
- Symptom date ≠ diagnosis onset

Note encounter dates:
{note_dates_json}

Per-note extractions:
{extractions_json}

OUTPUT — return ONLY this JSON array:
[
  {{
    "condition_name": "Left midline tongue carcinoma",
    "category": "cancer",
    "subcategory": "primary_malignancy",
    "status": "active",
    "onset": "May 2014",
    "evidence": [
      {{"note_id": "text_0", "line_no": 13, "span": "Left midline tongue carcinoma pT1 pN0"}}
    ]
  }}
]"""


def parse_retry_seconds(error_message: str) -> float:
    match = re.search(r'try again in (\d+(?:\.\d+)?)', str(error_message))
    if match:
        return float(match.group(1)) + 2
    return 60.0


def clean_json_response(raw: str) -> str:
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
    compressed = {}
    for note_id, conditions in all_extractions.items():
        compressed[note_id] = [
            {
                "condition_name": c.get("condition_name", ""),
                "section":        c.get("section", ""),
                "status_hint":    c.get("status_hint", ""),
                "onset_hint":     c.get("onset_hint", ""),
                "line_no":        c.get("line_no", 0),
                "span":           c.get("span", "")[:80]
            }
            for c in conditions
        ]
    return compressed


def call_merger_with_smart_retry(prompt: str, max_attempts: int = 5) -> str:
    for attempt in range(max_attempts):
        try:
            response = get_client().chat.completions.create(
                model=get_model(),
                messages=[
                    {"role": "system", "content": MERGER_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.0,
                max_tokens=4000
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            wait_secs = parse_retry_seconds(str(e))
            if attempt < max_attempts - 1:
                print(f"\n  Rate limit hit in merger, waiting {wait_secs:.0f}s...")
                time.sleep(wait_secs)
            else:
                raise
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(30)
            else:
                raise


def _merge_batch(extractions: dict, note_dates: dict) -> list:
    extractions_json = json.dumps(extractions, indent=1)
    note_dates_json  = json.dumps(note_dates)

    prompt = MERGER_USER_PROMPT.format(
        taxonomy=TAXONOMY_STR,
        disambiguation=DISAMBIGUATION_STR,
        extractions_json=extractions_json,
        note_dates_json=note_dates_json
    )

    approx_tokens = len(prompt) // 4
    print(f"  Batch prompt ~{approx_tokens} tokens")

    raw = call_merger_with_smart_retry(prompt)
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


def merge_and_classify(all_extractions: dict, note_dates: dict) -> list:
    
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
    dates1 = {k: note_dates[k] for k in note_ids[:mid] if k in note_dates}
    dates2 = {k: note_dates[k] for k in note_ids[mid:] if k in note_dates}

    print(f"  Batch 1: notes {note_ids[:mid]}")
    results1 = _merge_batch(batch1, dates1)
    time.sleep(10)

    print(f"  Batch 2: notes {note_ids[mid:]}")
    results2 = _merge_batch(batch2, dates2)
    time.sleep(10)

    print(f"  Final merge: combining {len(results1)} + {len(results2)} conditions...")
    combined = {"batch1": results1, "batch2": results2}
    return _merge_batch(combined, note_dates)