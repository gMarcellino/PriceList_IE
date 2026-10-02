import json
import os
from pathlib import Path
import sys

# Add workspace root to Python path for relative imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)
    
from src.entityRecognizer.LLMEntityRecognizer import ResponseParser, LLM_LOCATION_PLACEHOLDER

raw_path = Path("predictions/LLMEntityRecognizerTestSet/raw/entities_IT_huggingface_google_gemma-3-27b-it.json")
source_path = Path("data/annotations/groundTruth/entities_test.json")
out_path = Path("predictions/LLMEntityRecognizerTestSet/parsed/entities_IT_huggingface_google_gemma-3-27b-it.json")

raw = json.loads(raw_path.read_text(encoding="utf-8"))
source = json.loads(source_path.read_text(encoding="utf-8"))

parsed = {}
for record_id, raw_entry in raw.items():
    source_entry = source[record_id]
    text = source_entry["text"]

    output_entry = dict(source_entry)
    output_entry["entities"] = ResponseParser.parse(
        raw_entry.get("raw_response", ""),
        text,
        LLM_LOCATION_PLACEHOLDER,
    )
    parsed[record_id] = output_entry

out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=4), encoding="utf-8")