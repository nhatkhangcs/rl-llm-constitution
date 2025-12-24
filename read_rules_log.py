#!/usr/bin/env python3
"""
Helper script to read and analyze the discovered rules log file.

Usage:
    python read_rules_log.py <path_to_jsonl_file>
    python read_rules_log.py logs/discovered_rules_*.jsonl
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def read_rules_log(filepath: str) -> List[Dict]:
    """Read rules from JSONL file"""
    rules = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rules.append(json.loads(line))
    return rules


def analyze_rules(rules: List[Dict]) -> Dict:
    """Analyze the discovered rules"""
    analysis = {
        "total_entries": len(rules),
        "unique_rules": len(set(r["rule"]["rule_id"] for r in rules)),
        "by_category": defaultdict(int),
        "by_type": defaultdict(int),
        "by_confidence": {
            "high": 0,  # >= 0.8
            "medium": 0,  # 0.7-0.8
            "low": 0  # < 0.7
        }
    }
    
    for entry in rules:
        rule = entry["rule"]
        analysis["by_category"][rule["category"]] += 1
        analysis["by_type"][rule["rule_type"]] += 1
        
        conf = rule["confidence"]
        if conf >= 0.8:
            analysis["by_confidence"]["high"] += 1
        elif conf >= 0.7:
            analysis["by_confidence"]["medium"] += 1
        else:
            analysis["by_confidence"]["low"] += 1
    
    return analysis


def print_summary(analysis: Dict):
    """Print summary of rules"""
    print("=" * 60)
    print("Rules Log Summary")
    print("=" * 60)
    print(f"Total entries: {analysis['total_entries']}")
    print(f"Unique rules: {analysis['unique_rules']}")
    print(f"\nBy Category:")
    for category, count in sorted(analysis['by_category'].items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")
    print(f"\nBy Type:")
    for rule_type, count in sorted(analysis['by_type'].items()):
        print(f"  {rule_type}: {count}")
    print(f"\nBy Confidence:")
    print(f"  High (>=0.8): {analysis['by_confidence']['high']}")
    print(f"  Medium (0.7-0.8): {analysis['by_confidence']['medium']}")
    print(f"  Low (<0.7): {analysis['by_confidence']['low']}")
    print("=" * 60)


def print_rules(rules: List[Dict], limit: int = 10):
    """Print sample rules"""
    print(f"\nSample Rules (showing first {limit}):")
    print("-" * 60)
    for i, entry in enumerate(rules[:limit], 1):
        rule = entry["rule"]
        context = entry["context"]
        print(f"\n{i}. Rule ID: {rule['rule_id']}")
        print(f"   Type: {rule['rule_type']} | Category: {rule['category']}")
        print(f"   Confidence: {rule['confidence']:.3f} | Frequency: {rule['frequency']}")
        print(f"   Pattern: {rule['pattern'][:80]}...")
        print(f"   Prompt: {context['prompt']['text'][:80]}...")
        print(f"   Reward: {context['reward']['reward']:.3f} ({context['reward']['explanation']})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_rules_log.py <path_to_jsonl_file>")
        print("\nExample:")
        print("  python read_rules_log.py logs/discovered_rules_20240101_120000.jsonl")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    print(f"Reading rules from: {filepath}")
    rules = read_rules_log(filepath)
    
    if not rules:
        print("No rules found in file.")
        sys.exit(0)
    
    analysis = analyze_rules(rules)
    print_summary(analysis)
    
    # Show sample rules
    print_rules(rules, limit=10)
    
    # Option to save full report
    if len(sys.argv) > 2 and sys.argv[2] == "--save":
        output_file = filepath.replace(".jsonl", "_report.json")
        with open(output_file, 'w') as f:
            json.dump({
                "analysis": analysis,
                "all_rules": rules
            }, f, indent=2)
        print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    main()


