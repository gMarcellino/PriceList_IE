#!/usr/bin/env python3
"""
Wrapper script for LLM-based entity recognition.

This script provides a convenient entry point that delegates to the LLMEntityRecognizer module's built-in CLI.
For details on available options, run:

    python run_llm_entity_recognizer.py --help

Example usage:
    python run_llm_entity_recognizer.py --config scripts/configs/LLMEntityRecognizer/azure_gpt-4o-mini.yaml

"""

from src.entityRecognizer.LLMEntityRecognizer import parse_args, print_args, run_inference
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    print_args(args)
    run_inference(args)
