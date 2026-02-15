# Guardrail Reverse-Engineering Pipeline

Brief overview of the current pipeline.

## Run
```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config.yaml
```

## Pipeline (current)
Each **outer iteration** does:
1. Generate `10 harmful + 10 benign` seed prompts.
2. Query the target LLM for responses.
3. Infer a **minimal list of rules** from all prompt-response pairs.
4. Build one fixed synthetic edge-case eval set (`10 harmful + 10 benign`).

Then an **inner step loop** repeats:
1. Re-infer rules from the same 20 interactions.
2. Evaluate each rule on the synthetic set.
3. Use per-rule **accuracy** as reward.
4. Update the white-box rule generator with KL-regularized policy-gradient RL.
5. Stop early when RL loss plateaus (`patience` / `min_delta`).

When an iteration converges, its final rules are passed into the next iteration’s prompt-generation instruction to increase diversity.

## Reward and RL objective
- Rule reward: `accuracy = correct / total` on synthetic eval prompts.
- RL loss (per sample):  
  `loss = -(advantage * logp_policy) + kl_coef * (logp_policy - logp_ref)`

## Outputs
Generated artifacts in `logs/` include:
- `prompt_response_interactions_*.jsonl`
- `rule_rewards_*.jsonl`
- `discovered_rules_*.txt`
- `iteration_<N>_step_metrics.csv`
- `iteration_<N>_step_loss.png`
