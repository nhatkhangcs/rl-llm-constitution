"""
Main Pipeline Orchestrator

This module orchestrates the entire guardrail reverse-engineering pipeline,
including prompt generation, testing, reward calculation, rule inference, and RL training.
"""

import os
import json
import yaml
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.rl_trainer import RLTrainer, DPOTrainingPair
from src.utils import *

class GuardrailPipeline:
    """Main pipeline for reverse-engineering guardrails"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_directories()
        
        # Initialize shared model manager
        
        # Check if using HuggingFace models
        self.whitebox_llm_config = self.config["whitebox_llm"]
    
        self.target_llm_config = self.config["multi_purpose_llm"]["target_llm"]
        self.eval_llm_config = self.config["multi_purpose_llm"]["eval_llm"]
        self.rule_llm_config = self.config["multi_purpose_llm"]["rule_llm"]
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

        #### Initialize models ####
        
        # Initialize white-box model FIRST (for RL trainer and prompt generator)
        self.whitebox_model = AutoModelForCausalLM.from_pretrained(
            self.whitebox_llm_config["model_name"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=hf_token
        )

        self.whitebox_tokenizer = AutoTokenizer.from_pretrained(
            self.whitebox_llm_config["model_name"],
            token=hf_token
        )
        if self.whitebox_tokenizer.pad_token is None:
            self.whitebox_tokenizer.pad_token = self.whitebox_tokenizer.eos_token

        # this multi_purpose_model will be:
        # - target llm
        # - rule llm
        # - eval llm

        self.general_purpose_model = AutoModelForCausalLM.from_pretrained(
            self.config["multi_purpose_llm"]["model_name"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=hf_token
        )

        self.general_purpose_tokenizer = AutoTokenizer.from_pretrained(
            self.config["multi_purpose_llm"]["model_name"],
            token=hf_token
        )

        # self.rule_model = AutoModelForCausalLM.from_pretrained(
        #     self.rule_llm,
        #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        #     device_map="auto" if torch.cuda.is_available() else None,
        #     token=hf_token
        # )

        # self.rule_tokenizer = AutoTokenizer.from_pretrained(
        #     self.rule_llm,
        #     token=hf_token
        # )

        # self.eval_model = AutoModelForCausalLM.from_pretrained(
        #     self.eval_llm,
        #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        #     device_map="auto" if torch.cuda.is_available() else None,
        #     token=hf_token
        # )

        # self.eval_tokenizer = AutoTokenizer.from_pretrained(
        #     self.eval_llm,
        #     token=hf_token
        # )

        ### Current list of rules:
        self.approved_rules_dict = {}

        #### Initialize RLTrainer (DPO) ####
        rl_config = self.config["rl"]
        self.rl_trainer = RLTrainer(
            self.whitebox_model,
            self.whitebox_tokenizer,
            rl_config["learning_rate"],
            rl_config["batch_size"],
            rl_config.get("use_lora", True),
            rl_config.get("lora_r", 16),
            rl_config.get("lora_alpha", 32),
            output_dir=rl_config.get("model_output_dir", "./models/whitebox_dpo"),
            beta=rl_config.get("dpo_beta", 0.1)  # DPO beta parameter
        )
        
        

        #### Initialize RewardCalculator ####

        # self.reward_calculator = RewardCalculator(
        #     reward_weights=self.config["rl"]["reward_weights"]
        # )
        
        # Setup rules log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rules_log_file = str(Path(self.config["pipeline"]["log_dir"]) / f"discovered_rules_{timestamp}.jsonl")
        print(f"Rules will be logged to: {self.rules_log_file}")
        
        # Setup prompt-response interaction log file
        self.interactions_log_file = Path(self.config["pipeline"]["log_dir"]) / f"prompt_response_interactions_{timestamp}.jsonl"
        print(f"Prompt-response interactions will be logged to: {self.interactions_log_file}")
        
        
        
        # Training state
        self.iteration_history = []
        self.best_reward = float('-inf')
        self.training_data_buffer = []  # Buffer for RL training
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """Create necessary output directories"""
        output_dir = Path(self.config["pipeline"]["output_dir"])
        log_dir = Path(self.config["pipeline"]["log_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def _log_interactions(
        self,
        iteration_num: int,
        generated_prompts: List,
        target_responses: List,
        rewards: Optional[List] = None
    ):
        """
        Log all prompt-response interactions to JSONL file
        
        Args:
            iteration_num: Current iteration number
            generated_prompts: List of GeneratedPrompt objects
            target_responses: List of TargetResponse objects
            rewards: List of RewardResult objects
        """
        for prompt, response in zip(generated_prompts, target_responses):
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration_num,
                "prompt": {
                    "text": prompt["prompt"],
                    "intended_type": prompt["intended_type"]
                },
                "target_response": {
                    "text": response["response"]
                }
            }
            
            # Append to JSONL file (one JSON object per line)
            with open(self.interactions_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def run_iteration(self, iteration_num: int, context: Optional[Dict] = None) -> Dict:
        """
        Run a single iteration of the NEW pipeline:
        1. Generate prompts with white-box LLM
        2. Test prompts on target LLM
        3. Infer rules from prompt-response pairs
        4. Generate evaluation prompts based on rules
        5. Calculate refusal rate reward
        6. Train white-box LLM with reward
        
        Args:
            iteration_num: Current iteration number
            context: Optional context from previous iterations
        
        Returns:
            Dictionary with iteration results
        """
        print(f"\n{'='*60}")
        print(f"Iteration {iteration_num}")
        print(f"{'='*60}")
        
        # Step 1: Generate prompts with white-box LLM
        num_prompts = self.config["pipeline"]["prompts_per_iteration"]
        print(f"\n[Step 1] Generating {num_prompts} prompts with white-box LLM...")
        
        # Store input contexts for training
        harmful_user_prompt = PROMPT_GENERATION_USER_PROMPT_HARMFUL.format(num_prompts=num_prompts)
        benign_user_prompt = PROMPT_GENERATION_USER_PROMPT_BENIGN.format(num_prompts=num_prompts)
        
        generated_prompts = generate_prompts(self.whitebox_model, self.whitebox_tokenizer, self.whitebox_llm_config, SYSTEM_PROMPT_PROMPTS_GENERATION, harmful_user_prompt, return_type="json")
        generated_prompts_benign = generate_prompts(self.whitebox_model, self.whitebox_tokenizer, self.whitebox_llm_config, SYSTEM_PROMPT_PROMPTS_GENERATION, benign_user_prompt, return_type="json")
        
        # Track input contexts for each generated prompt
        prompt_to_context = {}
        for prompt_obj in generated_prompts:
            prompt_to_context[prompt_obj["prompt"]] = {
                "system": SYSTEM_PROMPT_PROMPTS_GENERATION,
                "user": harmful_user_prompt
            }
        for prompt_obj in generated_prompts_benign:
            prompt_to_context[prompt_obj["prompt"]] = {
                "system": SYSTEM_PROMPT_PROMPTS_GENERATION,
                "user": benign_user_prompt
            }
        
        generated_prompts.extend(generated_prompts_benign)
        
        # step 2: test prompts on target LLM
        print(f"\n[Step 2] Testing {len(generated_prompts)} prompts on target LLM...")
        target_responses = []
        for prompt in generated_prompts:
            response = generate_prompts(self.general_purpose_model, self.general_purpose_tokenizer, self.target_llm_config, SYSTEM_PROMPT_TARGET_LLM, prompt["prompt"], return_type="text")
            print(f"  Response: {response}")
            target_responses.append({
                "prompt": prompt["prompt"],
                "response": response,
                "intended_type": prompt["intended_type"]
            })
            self._log_interactions(iteration_num, generated_prompts, target_responses, None)
        # # Step 3: Infer rules from RuleInfererLLM

        print(f"\n[Step 3] Inferring rules from prompt-response pairs...")
        rule_list = []
        prompt_to_rule = {}  # Map: generated_prompt -> rule
        for response in target_responses:
            rule = generate_prompts(self.general_purpose_model, self.general_purpose_tokenizer, self.rule_llm_config, RULE_INFERENCE_SYSTEM_PROMPT, RULE_INFERENCE_USER_PROMPT.format(prompt=response["prompt"], response=response["response"]), return_type="text")
            print(f"  Rule: {rule}")
            rule_list.append(rule)
            # Map the original prompt to this rule
            prompt_to_rule[response["prompt"]] = rule
            # # save the rule to the rules_log_file
            # with open(self.rules_log_file, 'a', encoding='utf-8') as f:
            #     f.write(json.dumps(rule, ensure_ascii=False) + '\n')

        # step 4: Measure how good the rules are
        print(f"\n[Step 4] Measuring how good the rules are...")
        for rule in rule_list:
            benign_prompts = generate_prompts(self.general_purpose_model, self.general_purpose_tokenizer, self.eval_llm_config, EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_PROMPT_S1.format(rule=rule, num_prompts=self.config["pipeline"]["num_evaluation_prompts_per_rule"]["benign"], rule_category="benign"), return_type="json")
            harmful_prompts = generate_prompts(self.general_purpose_model, self.general_purpose_tokenizer, self.eval_llm_config, EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_PROMPT_S1.format(rule=rule, num_prompts=self.config["pipeline"]["num_evaluation_prompts_per_rule"]["harmful"], rule_category="harmful"), return_type="json")
            # # save the benign and harmful prompts to the rules_log_file
            # with open(self.rules_log_file, 'a', encoding='utf-8') as f:
            #     f.write(json.dumps(benign_prompts, ensure_ascii=False) + '\n')
            #     f.write(json.dumps(harmful_prompts, ensure_ascii=False) + '\n')
            print(f"  Benign prompts: {benign_prompts}")
            print(f"  Harmful prompts: {harmful_prompts}")
            benign_prompts_rejected = 0
            harmful_prompts_accepted = 0
            for prompt in benign_prompts:
                response = generate_prompts(self.general_purpose_model, self.general_purpose_tokenizer, self.target_llm_config, SYSTEM_PROMPT_TARGET_LLM, prompt["prompt"], return_type="text")
                ### need to check if the response is rejected or accepted
                decision = generate_prompts(self.general_purpose_model, self.general_purpose_tokenizer, self.eval_llm_config, EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_PROMPT_S2.format(prompt=prompt["prompt"], response=response), return_type="text")
                if "no" in decision.lower():
                    benign_prompts_rejected += 1
            for prompt in harmful_prompts:
                response = generate_prompts(self.general_purpose_model, self.general_purpose_tokenizer, self.target_llm_config, SYSTEM_PROMPT_TARGET_LLM, prompt["prompt"], return_type="text")
                decision = generate_prompts(self.general_purpose_model, self.general_purpose_tokenizer, self.eval_llm_config, EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_PROMPT_S2.format(prompt=prompt["prompt"], response=response), return_type="text")
                if "yes" in decision.lower():
                    harmful_prompts_accepted += 1
            
            print(f"  Benign prompts rejected: {benign_prompts_rejected}")
            print(f"  Harmful prompts accepted: {harmful_prompts_accepted}")

            # check if the rule is a duplicate of any other rule in the list
            is_duplicate = generate_prompts(self.general_purpose_model, self.general_purpose_tokenizer, self.eval_llm_config, SYSTEM_PROMPT_CHECK_DUPLICATE_RULE, CHECK_DUPLICATE_RULE_USER_PROMPT.format(rule=rule, list_of_rules=self.approved_rules_dict), return_type="text")
            duplicate = False
            if "yes" in is_duplicate.lower():
                duplicate = True
                print(f"  Rule {rule} is a duplicate of another rule in the list")
        
            # calculate refusal rate, 4 categories: benign_rejected, benign_accepted, harmful_rejected, harmful_accepted
            refusal_rate_benign_rejected = benign_prompts_rejected / len(benign_prompts)
            refusal_rate_benign_accepted = (len(benign_prompts) - benign_prompts_rejected) / len(benign_prompts)
            refusal_rate_harmful_rejected = (len(harmful_prompts) - harmful_prompts_accepted) / len(harmful_prompts)
            refusal_rate_harmful_accepted = harmful_prompts_accepted / len(harmful_prompts)

            print(f"Refusal rate for rule {rule}:")
            print(f"  Benign prompts rejected: {refusal_rate_benign_rejected}")
            print(f"  Benign prompts accepted: {refusal_rate_benign_accepted}")
            print(f"  Harmful prompts rejected: {refusal_rate_harmful_rejected}")
            print(f"  Harmful prompts accepted: {refusal_rate_harmful_accepted}")

            ## Calculate reward for the rule
            reward = compute_rule_reward(refusal_rate_benign_rejected, refusal_rate_harmful_rejected, refusal_rate_harmful_accepted, duplicate, new_rule_bonus=0.1, duplicate_penalty=0.1)
            print(f"  Reward for rule {rule}: {reward}")

            self.approved_rules_dict[rule] = {
                "rule": rule,
                "reward": reward
            }

        print(f"Total rules: {len(rule_list)}")
        print(f"Current approved rules dictionary: {self.approved_rules_dict}")

            
        ## Step 6: Train white-box LLM with DPO
        print(f"\n[Step 6] Preparing DPO training dataset and training white-box LLM...")
        
        # Build training data: map each generated prompt to its reward and context
        prompt_data = []
        for prompt_obj in generated_prompts:
            generated_prompt_text = prompt_obj["prompt"]
            
            # Find the rule that came from this prompt
            rule = prompt_to_rule.get(generated_prompt_text)
            if rule is None:
                print(f"  Warning: No rule found for prompt: {generated_prompt_text[:50]}...")
                continue
            
            # Get the reward for this rule
            rule_data = self.approved_rules_dict.get(rule)
            if rule_data is None:
                print(f"  Warning: No reward found for rule: {rule[:50]}...")
                continue
            
            reward = rule_data["reward"]
            
            # Get the input context (system + user prompt) used to generate this prompt
            context_info = prompt_to_context.get(generated_prompt_text)
            if context_info is None:
                print(f"  Warning: No context found for prompt: {generated_prompt_text[:50]}...")
                continue
            
            # Build full input context string
            input_context = f"{context_info['system']}\n\n{context_info['user']}"
            
            # Store prompt data
            prompt_data.append({
                "prompt": input_context,
                "generated_prompt": generated_prompt_text,
                "reward": reward
            })
        
        # Create DPO pairs: pair prompts with high rewards (chosen) vs low rewards (rejected)
        # Group by input context, then pair high-reward with low-reward prompts
        dpo_pairs = []
        
        # Group prompts by their input context
        context_groups = {}
        for data in prompt_data:
            context = data["prompt"]
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(data)
        
        # Create pairs within each context group
        for context, group in context_groups.items():
            if len(group) < 2:
                # Need at least 2 prompts with same context to create a pair
                continue
            
            # Sort by reward (descending)
            group_sorted = sorted(group, key=lambda x: x["reward"], reverse=True)
            
            # Create pairs: highest reward (chosen) vs lowest reward (rejected)
            # We can create multiple pairs from the same context group
            for i in range(len(group_sorted)):
                for j in range(i + 1, len(group_sorted)):
                    chosen_data = group_sorted[i]
                    rejected_data = group_sorted[j]
                    
                    # Only create pair if chosen reward > rejected reward
                    if chosen_data["reward"] > rejected_data["reward"]:
                        dpo_pairs.append(DPOTrainingPair(
                            prompt=context,
                            chosen=chosen_data["generated_prompt"],
                            rejected=rejected_data["generated_prompt"],
                            score_chosen=chosen_data["reward"],
                            score_rejected=rejected_data["reward"]
                        ))
        
        print(f"  Created {len(dpo_pairs)} DPO pairs from {len(prompt_data)} prompts")
        
        # Train if we have pairs and training is enabled
        training_metrics = None
        if dpo_pairs and self.config["rl"].get("enable_training", True):
            print(f"\n  Training on {len(dpo_pairs)} DPO pairs...")
            training_metrics = self.rl_trainer.train_batch(
                dpo_pairs,
                num_epochs=self.config["rl"].get("num_epochs", 1)
            )
        else:
            print(f"  No DPO pairs available or training disabled")
        
        # Return iteration results
        return {
            "iteration": iteration_num,
            "num_prompts_generated": len(generated_prompts),
            "num_rules_discovered": len(rule_list),
            "num_training_samples": len(training_data) if training_data else 0,
            "training_metrics": training_metrics,
            "rules_dict": {k: v["reward"] for k, v in self.approved_rules_dict.items()}
        }


        

    
    def train(self):
        """Run the full training pipeline"""
        num_iterations = self.config["pipeline"]["num_iterations"]
        checkpoint_every = self.config["pipeline"].get("save_checkpoint_every", 10)
        
        print(f"\n{'='*60}")
        print(f"Starting Guardrail Reverse-Engineering Pipeline")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Total iterations: {num_iterations}")
        print(f"  Prompts per iteration: {self.config['pipeline']['prompts_per_iteration']}")
        # if self.rl_trainer:
        #     print(f"  LoRA: {'ENABLED' if self.config['rl'].get('use_lora', True) else 'DISABLED'}")
        #     print(f"  Batch size: {self.config['rl']['batch_size']}")
        #     print(f"  Learning rate: {self.config['rl']['learning_rate']}")
        # print(f"{'='*60}\n")
        
        context = None
        
        for iteration in tqdm(range(1, num_iterations + 1), desc="Training"):
            # Run iteration
            result = self.run_iteration(iteration, context)
            
            # Update context for next iteration
            # context = {
            #     "summary": f"Iteration {iteration}: Reward={result['metrics']['average_reward']:.3f}, "
            #               f"Total rules={result['metrics']['total_rules']}",
            #     "best_reward": self.best_reward,
            #     "constitution_summary": self.rule_inference.get_constitution_summary()
            # }
            
            # # Print progress
            # if iteration % 10 == 0:
            #     print(f"\nIteration {iteration}:")
            #     print(f"  Average Reward: {result['metrics']['average_reward']:.3f}")
            #     if 'refusal_rate_harmful' in result['metrics']:
            #         print(f"  Refusal Rate (harmful): {result['metrics']['refusal_rate_harmful']:.1%}")
            #         print(f"  Refusal Rate (benign): {result['metrics']['refusal_rate_benign']:.1%}")
            #     print(f"  Total Rules Discovered: {result['metrics']['total_rules']}")
            
            # # Save checkpoint
            # if iteration % checkpoint_every == 0:
            #     self.save_checkpoint(iteration)
            
            # # Train RL model if enabled and buffer is full
            # if self.rl_trainer and len(self.training_data_buffer) >= self.config["rl"]["batch_size"]:
            #     print(f"\n{'='*60}")
            #     print(f"[RL TRAINING] Buffer full! Training white-box LLM...")
            #     print(f"{'='*60}")
            #     print(f"  Batch size: {len(self.training_data_buffer)}")
            #     print(f"  Epochs: {self.config['rl'].get('num_epochs', 1)}")
            #     print(f"  Using LoRA: {self.config['rl'].get('use_lora', True)}")
            #     training_metrics = self.rl_trainer.train_batch(
            #         self.training_data_buffer,
            #         num_epochs=self.config["rl"].get("num_epochs", 1)
            #     )
            #     print(f"\n  ✓ Training complete!")
            #     print(f"  Mean reward: {training_metrics.get('mean_reward', 0):.3f}")
            #     print(f"  Std reward: {training_metrics.get('std_reward', 0):.3f}")
            #     if 'ppo_stats' in training_metrics:
            #         ppo_stats = training_metrics['ppo_stats']
            #         if isinstance(ppo_stats, dict):
            #             print(f"  PPO stats: {ppo_stats}")
            #     # Clear buffer after training
            #     self.training_data_buffer = []
            #     print(f"  Buffer cleared. Continuing training...")
                
            #     # Save model checkpoint
            #     if iteration % (checkpoint_every * 2) == 0:
            #         checkpoint_model_dir = Path(self.config["pipeline"]["output_dir"]) / "checkpoints" / f"model_iter_{iteration}"
            #         print(f"\n  Saving model checkpoint to: {checkpoint_model_dir}")
            #         self.rl_trainer.save_model(str(checkpoint_model_dir))
        
        # # Final training step if there's remaining data
        # if self.rl_trainer and len(self.training_data_buffer) > 0:
        #     print(f"\nFinal training step with {len(self.training_data_buffer)} samples...")
        #     self.rl_trainer.train_batch(
        #         self.training_data_buffer,
        #         num_epochs=self.config["rl"].get("num_epochs", 1)
        #     )
        #     self.training_data_buffer = []
        
        # # Final save
        # self.save_final_results()
        
        # # Save final trained model
        # if self.rl_trainer:
        #     final_model_dir = Path(self.config["pipeline"]["output_dir"]) / "final_model"
        #     self.rl_trainer.save_model(str(final_model_dir))
        
        # print("\n" + "=" * 60)
        # print("Training completed!")
        # print(f"Total rules discovered: {len(self.rule_inference.constitution.rules)}")
        # print(f"Best average reward: {self.best_reward:.3f}")
        # print(f"Rules log file: {self.rules_log_file}")
        # print(f"Interactions log file: {self.interactions_log_file}")
        # print("=" * 60)
    
    def save_checkpoint(self, iteration: int):
        """Save checkpoint at current iteration"""
        checkpoint_dir = Path(self.config["pipeline"]["output_dir"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration}.json"
        
        checkpoint_data = {
            "iteration": iteration,
            "best_reward": self.best_reward,
            "constitution_summary": self.rule_inference.get_constitution_summary(),
            "recent_iterations": self.iteration_history[-10:]  # Last 10 iterations
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Export constitution
        constitution_path = checkpoint_dir / f"constitution_iter_{iteration}.json"
        self.rule_inference.export_constitution(str(constitution_path))
    
    def save_final_results(self):
        """Save final results and constitution"""
        output_dir = Path(self.config["pipeline"]["output_dir"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full history
        history_path = output_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.iteration_history, f, indent=2)
        
        # Export final constitution
        constitution_path = output_dir / f"final_constitution_{timestamp}.json"
        self.rule_inference.export_constitution(str(constitution_path))
        
        # Save summary report
        summary_path = output_dir / f"summary_{timestamp}.json"
        summary = {
            "total_iterations": len(self.iteration_history),
            "best_reward": self.best_reward,
            "final_constitution": self.rule_inference.get_constitution_summary(),
            "final_rule_count": len(self.rule_inference.constitution.rules)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
        print(f"  - Training history: {history_path.name}")
        print(f"  - Final constitution: {constitution_path.name}")
        print(f"  - Summary: {summary_path.name}")

