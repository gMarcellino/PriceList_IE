set -euo pipefail

## ENTITY RECOGNIZER ##
python3 run_llm_entity_recognizer.py --config scripts/configs/EntityRecognizer/LLMEntityRecognizer/azure_gpt-4o-mini.yaml
python3 run_llm_entity_recognizer.py --config scripts/configs/EntityRecognizer/LLMEntityRecognizer/azure_gpt-4o.yaml
python3 run_llm_entity_recognizer.py --config scripts/configs/EntityRecognizer/LLMEntityRecognizer/ollama_gemma-4b.yaml
python3 run_llm_entity_recognizer.py --config scripts/configs/EntityRecognizer/LLMEntityRecognizer/ollama_gemma-12b.yaml
python3 run_llm_entity_recognizer.py --config scripts/configs/EntityRecognizer/LLMEntityRecognizer/ollama_gemma-27b.yaml
