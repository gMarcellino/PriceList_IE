#!/usr/bin/env python3
"""
Wrapper script for GLiNER-based entity recognition.

This script provides a convenient entry point that delegates to the GLiNEREntityRecognizer module's built-in CLI.
For details on available options, run:

    python run_gliner_entity_recognizer.py --help

Example usage:
    python run_gliner_entity_recognizer.py --config scripts/configs/GLiNEREntityRecognizer/NuNerZero.yaml

"""

from src.entityRecognizer.GLiNEREntityRecognizer import parse_args, print_args, run_inference
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    print_args(args)
    run_inference(args)
