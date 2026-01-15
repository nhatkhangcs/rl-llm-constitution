"""
Main Pipeline Orchestrator

This module orchestrates the entire guardrail reverse-engineering pipeline,
including prompt generation, testing, reward calculation, rule inference, and RL training.
"""

import os
import gc
import json
import yaml
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.rl_trainer import RLTrainer, DPOTrainingPair
from src.utils import *

load_dotenv()

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
        self.hf_cache_dir = self._setup_hf_cache(self.config.get("hf_cache_dir"))
        
        # Initialize shared model manager
        self.model_output_dir = Path(self.config["rl"].get("model_output_dir", "./models/whitebox_dpo"))
        # Prefer bf16 for speed if supported; fallback to fp16 on CUDA, otherwise float32
        bf16_supported = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        self.whitebox_dtype = torch.bfloat16 if bf16_supported else torch.float16 if torch.cuda.is_available() else torch.float32
        self.general_dtype = torch.bfloat16 if bf16_supported else torch.float16 if torch.cuda.is_available() else torch.float32
        self.whitebox_use_bf16 = self.whitebox_dtype == torch.bfloat16
        self.whitebox_use_fp16 = self.whitebox_dtype == torch.float16
        
        # Check if using HuggingFace models
        self.whitebox_llm_config = self.config["whitebox_llm"]
    
        self.target_llm_config = self.config["multi_purpose_llm"]["target_llm"]
        self.eval_llm_config = self.config["multi_purpose_llm"]["eval_llm"]
        self.rule_llm_config = self.config["multi_purpose_llm"]["rule_llm"]
        self.hf_token = os.getenv("HF_TOKEN")

        print("Config: ", self.config)
        print("Whitebox LLM config: ", self.whitebox_llm_config)
        print("Target LLM config: ", self.target_llm_config)
        print("Eval LLM config: ", self.eval_llm_config)
        print("Rule LLM config: ", self.rule_llm_config)

        #### Initialize models ####
        
        # Get device IDs from config (relative to CUDA_VISIBLE_DEVICES)
        # If CUDA_VISIBLE_DEVICES=3,4, then device_id 0 = GPU 3, device_id 1 = GPU 4
        whitebox_device_id = self.whitebox_llm_config.get("device_id", 0)
        general_device_id = self.config["multi_purpose_llm"].get("device_id", 0)
        
        # Create explicit device_map for each model
        whitebox_device_map = f"cuda:{whitebox_device_id}" if torch.cuda.is_available() else None
        general_device_map = f"cuda:{general_device_id}" if torch.cuda.is_available() else None
        
        print(f"[Multi-GPU] White-box model will use device: {whitebox_device_id}")
        print(f"[Multi-GPU] General-purpose model will use device: {general_device_id}")
        
        # Initialize white-box model FIRST (for RL trainer and prompt generator)
        resume_checkpoint = self._find_latest_checkpoint(self.model_output_dir)
        self.resume_iteration = self._get_iteration_index(resume_checkpoint)
        whitebox_source = resume_checkpoint if resume_checkpoint else self.whitebox_llm_config["model_name"]
        if resume_checkpoint:
            print(f"[White-box] Resuming from checkpoint: {resume_checkpoint}")
            # Free any lingering GPU cache before loading checkpoint to avoid OOM
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            print(f"[White-box] Loading base model: {whitebox_source}")
        self.whitebox_model = AutoModelForCausalLM.from_pretrained(
            whitebox_source,
            torch_dtype=self.whitebox_dtype,
            device_map=whitebox_device_map,
            token=self.hf_token,
            cache_dir=self.hf_cache_dir
        )

        self.whitebox_tokenizer = AutoTokenizer.from_pretrained(
            whitebox_source,
            token=self.hf_token,
            cache_dir=self.hf_cache_dir
        )
        self.current_whitebox_source = str(whitebox_source)

        # this multi_purpose_model will be:
        # - target llm
        # - rule llm
        # - eval llm

        self.general_purpose_model = AutoModelForCausalLM.from_pretrained(
            self.config["multi_purpose_llm"]["model_name"],
            torch_dtype=self.general_dtype,
            device_map=general_device_map,
            token=self.hf_token,
            cache_dir=self.hf_cache_dir
        )

        self.general_purpose_tokenizer = AutoTokenizer.from_pretrained(
            self.config["multi_purpose_llm"]["model_name"],
            token=self.hf_token,
            cache_dir=self.hf_cache_dir
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
        
        # Setup wandb run name (same run for all iterations) - must be before RLTrainer init
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wandb_run_name = f"guardrail-reverse-eng-{timestamp}"
        print(f"Wandb run name: {self.wandb_run_name} (will reuse same run across all iterations)")

        #### Initialize RLTrainer (DPO) ####
        rl_config = self.config["rl"]
        self.rl_trainer = RLTrainer(
            self.whitebox_model,
            self.whitebox_tokenizer,
            rl_config["learning_rate"],
            rl_config["batch_size"],
            rl_config.get("gradient_accumulation_steps", 2),
            rl_config.get("max_length", 512),
            self.whitebox_use_bf16,
            self.whitebox_use_fp16,
            rl_config.get("gradient_checkpointing", True),
            rl_config.get("use_lora", True),
            rl_config.get("lora_r", 16),
            rl_config.get("lora_alpha", 32),
            output_dir=rl_config.get("model_output_dir", "./models/whitebox_dpo"),
            beta=rl_config.get("dpo_beta", 0.1),  # DPO beta parameter
            wandb_run_name=self.wandb_run_name  # Pass wandb run name for consistent logging
        )
        
        

        #### Initialize RewardCalculator ####

        # self.reward_calculator = RewardCalculator(
        #     reward_weights=self.config["rl"]["reward_weights"]
        # )
        
        # Setup rules log file paths (reuse timestamp from above)
        self.rules_log_file = str(Path(self.config["pipeline"]["log_dir"]) / f"discovered_rules_{timestamp}.jsonl")
        self.rules_txt_file = str(Path(self.config["pipeline"]["log_dir"]) / f"discovered_rules_{timestamp}.txt")
        self.rewards_log_file = Path(self.config["pipeline"]["log_dir"]) / f"rule_rewards_{timestamp}.jsonl"
        print(f"Rules will be logged to: {self.rules_log_file}")
        print(f"Rules will be saved to: {self.rules_txt_file}")
        print(f"Rule rewards will be logged to: {self.rewards_log_file}")
        
        # Setup prompt-response interaction log file
        self.interactions_log_file = Path(self.config["pipeline"]["log_dir"]) / f"prompt_response_interactions_{timestamp}.jsonl"
        print(f"Prompt-response interactions will be logged to: {self.interactions_log_file}")
        
        
        
        # Training state
        self.iteration_history = []
        self.best_reward = float('-inf')
        self.training_data_buffer = []  # Buffer for RL training

    def _find_latest_checkpoint(self, base_dir: Path) -> Optional[Path]:
        """
        Find the latest iteration checkpoint under the model output directory.
        Preference order:
        - Highest numbered iteration_* subdir
        - Otherwise, the base dir itself if it contains a saved model
        """
        if not base_dir.exists():
            return None
        iteration_dirs = [p for p in base_dir.glob("iteration_*") if p.is_dir()]
        if iteration_dirs:
            try:
                for candidate in sorted(iteration_dirs, key=lambda p: int(p.name.split("_")[-1]), reverse=True):
                    if self._is_valid_checkpoint_dir(candidate):
                        return candidate
            except Exception:
                pass
        # If the base dir has model files, allow loading from there
        return base_dir if self._is_valid_checkpoint_dir(base_dir) else None

    def _get_iteration_index(self, checkpoint_path: Optional[Path]) -> int:
        """Extract iteration index from checkpoint path name (iteration_N). Base dir = 0."""
        if checkpoint_path is None:
            return 0
        name = checkpoint_path.name
        if name.startswith("iteration_"):
            try:
                return int(name.split("_")[-1])
            except Exception:
                return 0
        return 0

    def _is_valid_checkpoint_dir(self, path: Path) -> bool:
        """A checkpoint is valid if it has a config.json (or adapter_config.json for PEFT)."""
        return (path / "config.json").exists() or (path / "adapter_config.json").exists()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_hf_cache(self, configured_cache_dir: Optional[str]) -> str:
        """
        Configure the HuggingFace cache directory with a configurable path.
        Priority: config value -> HF_CACHE_DIR env -> default ~/.cache/huggingface/hub
        """
        cache_dir = configured_cache_dir or os.getenv("HF_CACHE_DIR")
        if cache_dir is None:
            cache_dir = "~/.cache/huggingface/hub"
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)

        os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
        os.environ.setdefault("HF_HOME", str(cache_path.parent))
        return str(cache_path)
    
    def setup_directories(self):
        """Create necessary output directories"""
        log_dir = Path(self.config["pipeline"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def _log_interactions(
        self,
        iteration_num: int,
        generated_prompts: List,
        target_responses: List,
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

    def _log_rule_reward(
        self,
        iteration_num: int,
        rule: str,
        reward: float,
        duplicate: bool,
        refusal_rates: Dict[str, float],
        counts: Dict[str, int],
    ):
        """Log per-rule reward details to JSONL for later analysis."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration_num,
            "rule": rule,
            "reward": reward,
            "duplicate": duplicate,
            "refusal_rates": refusal_rates,
            "counts": counts,
        }
        with open(self.rewards_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def _save_rules_to_txt(self):
        """
        Save discovered rules to a human-readable txt file
        
        The file will contain:
        - Total number of rules
        - Each rule with its reward score, sorted by reward (highest first)
        """
        if not self.approved_rules_dict:
            print("  No rules to save")
            return
        
        # Sort rules by reward (descending)
        sorted_rules = sorted(
            self.approved_rules_dict.items(),
            key=lambda x: x[1]["reward"],
            reverse=True
        )
        
        # Write to txt file
        with open(self.rules_txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DISCOVERED RULES\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Rules Discovered: {len(sorted_rules)}\n")
            f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, (rule_text, rule_data) in enumerate(sorted_rules, 1):
                reward = rule_data["reward"]
                f.write(f"Rule #{idx}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Reward Score: {reward:.4f}\n")
                f.write(f"Rule Text: {rule_text}\n")
                f.write("\n")
        
        print(f"  ✓ Saved {len(sorted_rules)} rules to {self.rules_txt_file}")
    
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

        ## Clear GPU memory and load latest checkpoint
        print(f"\n[Loading] Preparing to load white-box model for iteration {iteration_num}...")
        
        # Step 1: Find latest checkpoint (if any)
        model_output_dir = Path(self.config["rl"]["model_output_dir"])
        latest_checkpoint = None
        
        if model_output_dir.exists():
            iteration_dirs = [
                d for d in model_output_dir.iterdir()
                if d.is_dir() and d.name.startswith("iteration_") and self._is_valid_checkpoint_dir(d)
            ]
            if iteration_dirs:
                def get_iteration_num(path):
                    try:
                        return int(path.name.split("_")[1])
                    except Exception:
                        return -1
                iteration_dirs.sort(key=get_iteration_num, reverse=True)
                latest_checkpoint = iteration_dirs[0]
                print(f"  Found latest checkpoint: {latest_checkpoint.name}")
            else:
                print(f"  No valid iteration checkpoints found; will load base model")
        
        desired_source = str(latest_checkpoint) if latest_checkpoint else self.whitebox_llm_config["model_name"]
        need_reload = (
            self.whitebox_model is None or
            getattr(self, "current_whitebox_source", None) != desired_source
        )
        
        if not need_reload:
            print(f"  Reusing already loaded white-box model: {self.current_whitebox_source}")
        else:
            # Step 2: Clear previous white-box model from GPU memory
            if hasattr(self, 'whitebox_model') and self.whitebox_model is not None:
                print(f"  Clearing previous white-box model from GPU...")
                # Break references held by RLTrainer and its DPOTrainer
                if hasattr(self, "rl_trainer") and self.rl_trainer is not None:
                    try:
                        self.rl_trainer.model = None
                        if hasattr(self.rl_trainer, "dpo_trainer") and self.rl_trainer.dpo_trainer is not None:
                            self.rl_trainer.dpo_trainer = None
                    except Exception:
                        pass
                del self.whitebox_model
                self.whitebox_model = None
            
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  ✓ GPU cache cleared (general_purpose_model still in memory)")
        
        # Step 3: Load model from checkpoint or base model
        # Get device ID for white-box model (relative to CUDA_VISIBLE_DEVICES)
        whitebox_device_id = self.whitebox_llm_config.get("device_id", 0)
        whitebox_device_map = f"cuda:{whitebox_device_id}" if torch.cuda.is_available() else None
        
        if need_reload:
            if latest_checkpoint and latest_checkpoint.exists():
                checkpoint_path = str(latest_checkpoint)
                print(f"  Loading white-box model from checkpoint: {checkpoint_path}")
                self.whitebox_model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=self.whitebox_dtype,
                    device_map=whitebox_device_map,
                    token=self.hf_token
                )
                tokenizer_path = checkpoint_path
                if not (latest_checkpoint / "tokenizer_config.json").exists():
                    tokenizer_path = self.whitebox_llm_config["model_name"]
                    print(f"  Tokenizer not found in checkpoint, using base model tokenizer")

                if RLTrainer._model_has_invalid_weights(self.whitebox_model):
                    print(f"  ⚠️ Checkpoint appears corrupted (NaN/Inf weights). Falling back to base model.")
                    self.whitebox_model = AutoModelForCausalLM.from_pretrained(
                        self.whitebox_llm_config["model_name"],
                        torch_dtype=self.whitebox_dtype,
                        device_map=whitebox_device_map,
                        token=self.hf_token,
                        cache_dir=self.hf_cache_dir
                    )
                    tokenizer_path = self.whitebox_llm_config["model_name"]
            else:
                print(f"  No checkpoint found, loading base model: {self.whitebox_llm_config['model_name']}")
                self.whitebox_model = AutoModelForCausalLM.from_pretrained(
                    self.whitebox_llm_config["model_name"],
                    torch_dtype=self.whitebox_dtype,
                    device_map=whitebox_device_map,
                    token=self.hf_token,
                    cache_dir=self.hf_cache_dir
                )
                tokenizer_path = self.whitebox_llm_config["model_name"]
                if (Path(model_output_dir) / "tokenizer_config.json").exists():
                    tokenizer_path = str(model_output_dir)
            self.current_whitebox_source = desired_source
        else:
            tokenizer_path = self.current_whitebox_source
        
        # Load tokenizer (tokenizers don't use torch_dtype or device_map)
        # Only use cache_dir if loading from model name (not checkpoint)
        tokenizer_kwargs = {"token": self.hf_token}
        if tokenizer_path == self.whitebox_llm_config["model_name"]:
            tokenizer_kwargs["cache_dir"] = self.hf_cache_dir
        self.whitebox_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs
        )
        if self.whitebox_tokenizer.pad_token is None:
            self.whitebox_tokenizer.pad_token = self.whitebox_tokenizer.eos_token
        
        print(f"  ✓ White-box model and tokenizer loaded")
        
        # Step 4: Update RLTrainer with new model
        print(f"  Updating RLTrainer with new model...")
        self.rl_trainer.model = self.whitebox_model
        self.rl_trainer.tokenizer = self.whitebox_tokenizer
        self.rl_trainer.device = next(self.whitebox_model.parameters()).device
        
        # If using LoRA, check if the loaded model already has LoRA adapters
        # Models saved with LoRA will have adapters, base models won't
        if self.config["rl"].get("use_lora", True):
            from peft import PeftModel
            # Check if model is already a PEFT model (has adapters)
            is_peft_model = isinstance(self.whitebox_model, PeftModel) or hasattr(self.whitebox_model, 'peft_config')
            
            if not is_peft_model:
                # Model doesn't have LoRA adapters, apply them
                print(f"  Applying LoRA adapters to loaded model...")
                from peft import get_peft_model, LoraConfig, TaskType
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config["rl"].get("lora_r", 16),
                    lora_alpha=self.config["rl"].get("lora_alpha", 32),
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )
                self.whitebox_model = get_peft_model(self.whitebox_model, lora_config)
                self.rl_trainer.model = self.whitebox_model
                print(f"  ✓ LoRA adapters applied")
            else:
                print(f"  ✓ Model already has LoRA adapters loaded from checkpoint")
        
        # Reinitialize DPO trainer if it exists (will be recreated on first train_batch call)
        if hasattr(self.rl_trainer, 'dpo_trainer'):
            self.rl_trainer.dpo_trainer = None
        print(f"  ✓ RLTrainer updated")

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
            self._log_interactions(iteration_num, generated_prompts, target_responses)
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
            reward = compute_rule_reward(
                refusal_rate_benign_rejected,
                refusal_rate_harmful_rejected,
                refusal_rate_harmful_accepted,
                duplicate,
                shaping_mode=self.config["rl"].get("shaping_mode", "linear"),
                new_rule_bonus=self.config["rl"].get("new_rule_bonus", 0.05),
                duplicate_penalty=self.config["rl"].get("duplicate_penalty", 0.4),
                w_sound_hr=self.config["rl"].get("w_sound_hr", 1.2),
                w_sound_br=self.config["rl"].get("w_sound_br", 1.0),
                w_sound_ha=self.config["rl"].get("w_sound_ha", 1.0),
                w_complete_hr=self.config["rl"].get("w_complete_hr", 0.6),
                w_margin_hr=self.config["rl"].get("w_margin_hr", 1.0),
                w_margin_br=self.config["rl"].get("w_margin_br", 1.0),
                w_margin_ha=self.config["rl"].get("w_margin_ha", 1.0),
                w_quad_hr=self.config["rl"].get("w_quad_hr", 1.2),
                w_quad_ha=self.config["rl"].get("w_quad_ha", 1.0),
                w_quad_br=self.config["rl"].get("w_quad_br", 1.5),
                w_br_penalty=self.config["rl"].get("w_br_penalty", 1.6)
            )
            print(f"  Reward for rule {rule}: {reward}")
            self._log_rule_reward(
                iteration_num=iteration_num,
                rule=rule,
                reward=reward,
                duplicate=duplicate,
                refusal_rates={
                    "benign_rejected": refusal_rate_benign_rejected,
                    "benign_accepted": refusal_rate_benign_accepted,
                    "harmful_rejected": refusal_rate_harmful_rejected,
                    "harmful_accepted": refusal_rate_harmful_accepted,
                },
                counts={
                    "benign_prompts_rejected": benign_prompts_rejected,
                    "benign_prompts_total": len(benign_prompts),
                    "harmful_prompts_accepted": harmful_prompts_accepted,
                    "harmful_prompts_total": len(harmful_prompts),
                },
            )

            self.approved_rules_dict[rule] = {
                "rule": rule,
                "reward": reward
            }

        print(f"Total rules: {len(rule_list)}")
        print(f"Current approved rules dictionary: {self.approved_rules_dict}")
        
        # Save rules to txt file after each iteration
        self._save_rules_to_txt()

            
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
            # Save the updated model after each iteration to allow resume
            iteration_model_dir = self.model_output_dir / f"iteration_{iteration_num}"
            print(f"  Saving iteration checkpoint to {iteration_model_dir}")
            self.rl_trainer.save_model(str(iteration_model_dir))
        else:
            print(f"  No DPO pairs available or training disabled")
        
        # Return iteration results
        return training_metrics


        

    
    def train(self):
        """Run the full training pipeline"""
        num_iterations = self.config["pipeline"]["num_iterations"]
        start_iteration = self.resume_iteration + 1 if self.resume_iteration else 1
        if start_iteration > num_iterations:
            print(f"\n[Resume] Latest checkpoint iteration_{self.resume_iteration} >= configured iterations ({num_iterations}); nothing to train.")
            return
        
        print(f"\n{'='*60}")
        print(f"Starting Guardrail Reverse-Engineering Pipeline")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Total iterations: {num_iterations}")
        print(f"  Prompts per iteration: {self.config['pipeline']['prompts_per_iteration']}")
        if self.resume_iteration:
            print(f"  Resuming from iteration {self.resume_iteration}, will run {num_iterations - self.resume_iteration} more iteration(s)")
        # if self.rl_trainer:
        #     print(f"  LoRA: {'ENABLED' if self.config['rl'].get('use_lora', True) else 'DISABLED'}")
        #     print(f"  Batch size: {self.config['rl']['batch_size']}")
        #     print(f"  Learning rate: {self.config['rl']['learning_rate']}")
        # print(f"{'='*60}\n")
        
        context = None
        
        for iteration in tqdm(range(start_iteration, num_iterations + 1), desc="Training"):
            # Run iteration
            result = self.run_iteration(iteration, context)
            print(f"  Training metrics: {result}")
            
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
                
    
