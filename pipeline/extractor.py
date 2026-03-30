import os
import json
import time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

def get_client():
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY")
    )

def get_model():
    return os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile")


EXTRACTION_SYSTEM_PROMPT = """Extract medical conditions. Return JSON array only. No markdown."""

EXTRACTION_USER_PROMPT = """List every medical condition in this clinical note excerpt.

INCLUDE:
- Named diagnoses (e.g. "tongue carcinoma", "arterial hypertension", "liver cirrhosis")
- Suspected conditions ("rule out", "possible", "suspected", "suggestive of")
- Past conditions ("history of", "status post")
- Specific named findings (named metastases, named fractures, named varices)

EXCLUDE — be strict about these:
- Symptoms alone: pain, fatigue, nausea, fever, swelling, bleeding
- Lab values: hemoglobin levels, creatinine values, CRP values
- Negative findings: "no evidence of", "no signs of", "unremarkable", "within normal"
- Vague words: "tumor", "lesion", "mass" without a specific diagnosis name
- Medications and dosages
- Procedures: surgery, biopsy, gastroscopy, kyphoplasty, resection
- Normal examination findings
- Elevated/reduced lab markers alone (e.g. "elevated LDH", "reduced hemoglobin")

A clinical note with 1-2 pages should have at most 10-15 conditions.
Be conservative — only extract named, confirmed or suspected diagnoses.

For each condition return:
- condition_name: specific diagnosis name, max 80 chars
- section: which section it is in (e.g. "Diagnoses", "Medical History", "CT Findings")
- status_hint: signal words, max 40 chars (e.g. "in Diagnoses section", "history of")
- onset_hint: date only, max 20 chars (e.g. "05/2014", "2018", "unknown")
- line_no: integer line number from the note
- span: short identifying phrase, max 60 chars

Return ONLY a JSON array [ ... ]. Nothing else.

NOTE EXCERPT:
{chunk_text}"""

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



def split_into_chunks(numbered_text: str, chunk_size: int = 100) -> list:
    
    lines = numbered_text.split("\n")
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i: i + chunk_size]
        chunks.append("\n".join(chunk_lines))
    return chunks


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=60)
)
def extract_from_chunk(chunk_text: str, note_id: str) -> list:

    non_empty = [l for l in chunk_text.split("\n") if l.strip()]
    if len(non_empty) < 3:
        return []

    prompt = EXTRACTION_USER_PROMPT.format(chunk_text=chunk_text)

    response = get_client().chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=800
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
            f"Chunk parse failed for {note_id}: {e}\n"
            f"Raw: {raw[:300]}"
        )


def extract_from_note(note: dict) -> list:
    
    chunks = split_into_chunks(note["numbered_text"], chunk_size=100)
    all_conditions = []

    for i, chunk in enumerate(chunks):
        conditions = extract_from_chunk(chunk, note["note_id"])
        all_conditions.extend(conditions)
        if i < len(chunks) - 1:
            time.sleep(2)

    seen_lines = set()
    unique_conditions = []
    for c in all_conditions:
        line_no = c.get("line_no")
        if line_no not in seen_lines:
            seen_lines.add(line_no)
            unique_conditions.append(c)

    for c in unique_conditions:
        c["source_note_id"]      = note["note_id"]
        c["note_encounter_date"] = note["encounter_date"]

    return unique_conditions