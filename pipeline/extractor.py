import os
import json
import time
import re
from openai import OpenAI, RateLimitError

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

EXCLUDE — be strict:
- Symptoms alone: pain, fatigue, nausea, fever, swelling, bleeding
- Lab values and lab markers alone (e.g. "elevated LDH", "hemoglobin 8.2")
- Negative findings: "no evidence of", "no signs of", "unremarkable"
- Vague words: "tumor", "lesion", "mass" without a specific diagnosis name
- Medications and dosages
- Procedures: surgery, biopsy, gastroscopy, kyphoplasty, resection
- Normal examination findings

Be conservative — a 1-2 page note should have at most 10-15 conditions.

For each condition return:
- condition_name: specific diagnosis name, max 80 chars
- section: which section (e.g. "Diagnoses", "Medical History", "CT Findings")
- status_hint: signal words, max 40 chars (e.g. "in Diagnoses section", "history of")
- onset_hint: date only, max 20 chars (e.g. "05/2014", "unknown")
- line_no: integer line number
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


def parse_retry_seconds(error_message: str) -> float:
    
    match = re.search(r'try again in (\d+(?:\.\d+)?)', str(error_message))
    if match:
        return float(match.group(1)) + 2  # add 2s buffer
    return 60.0  # default wait 60s if we can't parse


def call_with_smart_retry(prompt: str, note_id: str, max_attempts: int = 5) -> str:
    
    for attempt in range(max_attempts):
        try:
            response = get_client().chat.completions.create(
                model=get_model(),
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.0,
                max_tokens=800
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            wait_secs = parse_retry_seconds(str(e))
            if attempt < max_attempts - 1:
                print(f"\n  Rate limit hit for {note_id}, waiting {wait_secs:.0f}s...")
                time.sleep(wait_secs)
            else:
                raise
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(30)
            else:
                raise


def split_into_chunks(numbered_text: str, chunk_size: int = 100) -> list:
    lines = numbered_text.split("\n")
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i: i + chunk_size]
        chunks.append("\n".join(chunk_lines))
    return chunks


def extract_from_chunk(chunk_text: str, note_id: str) -> list:
    non_empty = [l for l in chunk_text.split("\n") if l.strip()]
    if len(non_empty) < 3:
        return []

    prompt = EXTRACTION_USER_PROMPT.format(chunk_text=chunk_text)
    raw = call_with_smart_retry(prompt, note_id)
    cleaned = clean_json_response(raw)

    try:
        conditions = json.loads(cleaned)
        if not isinstance(conditions, list):
            return []
        return conditions
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse failed for chunk of {note_id}, skipping")
        return []


def extract_from_note(note: dict) -> list:
    """
    Extracts all conditions from a note by processing 100-line chunks.
    Combines results and deduplicates by line number.
    """
    chunks = split_into_chunks(note["numbered_text"], chunk_size=100)
    all_conditions = []

    for i, chunk in enumerate(chunks):
        conditions = extract_from_chunk(chunk, note["note_id"])
        all_conditions.extend(conditions)
        if i < len(chunks) - 1:
            time.sleep(6)

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