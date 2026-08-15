#!/bin/bash
# Prompt-free wrapper for checkpoint_metrics.py (read-only, zero API calls).
cd "$(dirname "$0")" && exec ../khra_dynamic+adversarial_code/.venv/bin/python checkpoint_metrics.py "$@"
