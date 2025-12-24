# Guardrail Reverse-Engineering Pipeline

This pipeline reverse-engineers the guardrails of a target LLM by using a white-box LLM to generate test prompts and discover safety rules through reinforcement learning.

# Training command
```bash
python main.py --config config.yaml --iterations 10
```

# How it works
1. Generate prompts with white-box LLM
2. Test prompts on target LLM
3. Infer rules from prompt-response pairs
4. Generate evaluation prompts based on rules
5. Calculate refusal rate reward
6. Train white-box LLM with reward
