#!/usr/bin/env python3
"""
Main Entry Point for Guardrail Reverse-Engineering Pipeline

This script runs the complete pipeline to reverse-engineer guardrails
of a target LLM by using a white-box LLM to generate test prompts.
"""

import argparse
import os
import sys
from pathlib import Path

from src.pipeline import GuardrailPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Reverse-engineer guardrails of a target LLM"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override number of iterations from config"
    )
    parser.add_argument(
        "--prompts-per-iteration",
        type=int,
        help="Override prompts per iteration from config"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"ERROR: Configuration file '{args.config}' not found.")
        print("Please create a config.yaml file or specify a different config path.")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = GuardrailPipeline(config_path=args.config)
    
    # Run training
    pipeline.train()
        

if __name__ == "__main__":
    main()


