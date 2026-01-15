#!/usr/bin/env python3
"""
Evaluation Script for Guardrail Discovery Pipeline

This script runs the evaluation pipeline to test rule discovery quality:
1. Loads ground-truth rules from rule bank
2. Samples a subset and implants them into target model
3. Runs discovery pipeline
4. Computes precision/recall metrics
5. Evaluates refusal rates for discovered vs ground-truth rules
"""

import argparse
import sys
from pathlib import Path

from src.evaluator import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate guardrail discovery pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to pipeline configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--rule-bank",
        type=str,
        default="rule_bank.txt",
        help="Path to rule bank file (default: rule_bank.txt)"
    )
    parser.add_argument(
        "--num-rules",
        type=int,
        default=5,
        help="Number of ground-truth rules to implant (default: 5)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of discovery iterations to run (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for rule sampling (default: None)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_output",
        help="Output directory for evaluation results (default: ./evaluation_output)"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"ERROR: Configuration file '{args.config}' not found.")
        sys.exit(1)
    
    # Check if rule bank file exists
    if not Path(args.rule_bank).exists():
        print(f"ERROR: Rule bank file '{args.rule_bank}' not found.")
        sys.exit(1)
    
    # Initialize evaluation pipeline
    print(f"[Evaluation] Initializing evaluation pipeline...")
    print(f"  Config: {args.config}")
    print(f"  Rule bank: {args.rule_bank}")
    print(f"  Rules to implant: {args.num_rules}")
    print(f"  Discovery iterations: {args.iterations}")
    print(f"  Output directory: {args.output_dir}")
    
    evaluator = EvaluationPipeline(
        config_path=args.config,
        rule_bank_path=args.rule_bank,
        num_rules_to_implant=args.num_rules,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(num_iterations=args.iterations)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Precision: {results['matching_metrics']['precision']:.3f}")
    print(f"Recall: {results['matching_metrics']['recall']:.3f}")
    print(f"F1 Score: {results['matching_metrics']['f1']:.3f}")
    print(f"\nDiscovered {results['matching_metrics']['num_discovered_rules']} rules")
    print(f"Matched {results['matching_metrics']['num_matched_discovered']} discovered rules")
    print(f"Matched {results['matching_metrics']['num_matched_ground_truth']}/{len(results['implanted_rules'])} implanted rules")
    print(f"{'='*60}\n")
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
