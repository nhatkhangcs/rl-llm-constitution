# Guardrail Reverse-Engineering Pipeline

This pipeline reverse-engineers the guardrails of a target LLM by using a white-box LLM to generate test prompts and discover safety rules through reinforcement learning.

# Training command
```bash
python main.py
```

# How it works
1. Generate prompts with white-box LLM
2. Test prompts on target LLM
3. Infer rules from prompt-response pairs
4. Generate evaluation prompts based on rules
5. Calculate refusal rate reward
6. Train white-box LLM with reward

# Configs
 You can configure the pipeline in the config.yaml file.
 - whitebox_llm: Configuration for the white-box LLM
 - multi_purpose_llm: Configuration for the multi-purpose LLM
 - rl: Configuration for the reinforcement learning
 - rule_inference: Configuration for the rule inference
 - pipeline: Configuration for the pipeline

# Outcome Rules
 The outcome rules will be saved in the logs directory as discovered_rules_{timestamp}.txt.