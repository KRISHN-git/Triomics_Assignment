#  LLM-Based Clinical Condition Extraction  
### From Longitudinal Patient Notes

## Overview
Clinical notes written by physicians are rich sources of medical information, but they are unstructured and difficult to analyse programmatically.

This project builds an automated pipeline that reads longitudinal patient notes — multiple markdown files per patient written over months or years — and extracts a structured inventory of every medical condition mentioned across all notes.

Each condition is extracted with:
- Human-readable name
- Taxonomy classification (category and subcategory)
- Status (active, resolved, suspected)
- Onset date
- Evidence from notes

The pipeline is powered by a Large Language Model (LLM) via an OpenAI-compatible API.

---

## System Architecture

1. Loader → Reads notes and numbers lines  
2. Extractor → Extracts conditions using LLM  
3. Merger → Deduplicates and classifies  
4. Validator → Ensures schema correctness  

Entrypoint: `main.py`

---

## Setup

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file:

```
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=llama-3.3-70b-versatile
```

---

## Run

```bash
python main.py --data-dir ./data/dev --patient-list ./patients.json --output-dir ./output --verbose
```
```bash
If "Too many requests" error occurs, then:

1. Get new Groq key.
- Go to https://console.groq.com
- Sign in with a different Google account or email.
- Click API Keys → Create API Key.
- Copy the new key (starts with gsk_...).

2. Update .env file.
3. Run the script again with --skip at the end.

 - python main.py --data-dir ./data/dev --patient-list ./patients.json --output-dir ./output --verbose --skip-done
```

---

## 📁 Project Structure

```
Triomics-medical_extraction/
  data/
    dev
    train
  output
  output_train
  pipeline/
    extractor.py
    merger.py
    validator.py
  utils/
    loader.py
  .gitignore
  evaluate.py
  main.py
  patients_train.json
  patients.json
  requirements.txt
  taxonomy.json
  test_full.py
```

---

## Key Features

- Chunked LLM processing
- Smart retry for rate limits
- Hierarchical merging to avoid token overflow
- Strict validation layer
- Model-agnostic design

---

## Challenges Solved

- API truncation issues
- Token overflow errors
- Rate limiting (429)
- Over-extraction by LLM

---

## Results

- Reliable extraction across multiple patient notes
- Improved classification using taxonomy
- Reduced noise via strict prompts

---

## Conclusion

This project demonstrates that a two-stage LLM pipeline — per-note extraction followed by cross-note merging and classification — can effectively extract structured medical condition inventories from unstructured clinical notes. The key engineering challenges were managing LLM API constraints: output truncation, per-minute and daily token limits, and token overflow in large prompts. 
