"""
Reinforcement Learning Trainer for White-box LLM

Supports:
- DPO (existing path)
- KL-regularized policy-gradient updates over scalar rewards (new path)
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
)
from transformers.training_args import ParallelMode
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from tqdm.auto import tqdm


@dataclass
class DPOTrainingPair:
    """Represents a DPO training pair (chosen vs rejected)"""
    def __init__(
        self,
        prompt: str,  # Input context (system + user prompt)
        chosen: str,  # Generated prompt with high reward
        rejected: str,  # Generated prompt with low reward
        score_chosen: float,  # Reward for chosen prompt
        score_rejected: float  # Reward for rejected prompt
    ):
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected
        self.score_chosen = score_chosen
        self.score_rejected = score_rejected


@dataclass
class RuleRLSample:
    """Single supervised-on-policy sample for KL-regularized policy-gradient."""
    prompt: str
    completion: str
    reward: float



class RLTrainer:
    """Trains white-box LLM with DPO and KL-regularized policy-gradient."""
    
    def __init__(
        self,
        model,
        tokenizer,
        learning_rate,
        batch_size,
        gradient_accumulation_steps,
        max_length,
        use_bf16,
        use_fp16,
        gradient_checkpointing,
        use_lora,
        lora_r,
        lora_alpha,
        output_dir: str = "./models/whitebox_dpo",
        beta: float = 0.1,  # DPO beta parameter (temperature)
        wandb_run_name: Optional[str] = None,  # Optional experiment name (no wandb by default)
        ref_model_path: Optional[str] = None,  # Path/repo-id for frozen reference model
        hf_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        ref_device: Optional[str] = None,
    ):
        """
        Initialize DPO Trainer
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            output_dir: Directory to save trained models
            beta: DPO beta parameter (controls strength of KL penalty)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.max_length = int(max_length)
        self.use_bf16 = bool(use_bf16)
        self.use_fp16 = bool(use_fp16)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.output_dir = output_dir
        self.beta = float(beta) if not isinstance(beta, float) else beta
        self.wandb_run_name = wandb_run_name
        self.ref_model_path = ref_model_path
        self.hf_token = hf_token
        self.hf_cache_dir = hf_cache_dir
        self.ref_device = ref_device
        
        print(f"[DPO Trainer] Model loaded")
        
        # Apply LoRA if requested (DPOTrainer will handle this via peft_config)
        peft_config = None
        if use_lora:
            print(f"[DPO Trainer] LoRA will be applied (r={lora_r}, alpha={lora_alpha})...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Common for Llama models
            )
        
        # Initialize DPO config
        # Use the already-converted values (learning_rate, batch_size, and beta are converted above)
        self.dpo_config = DPOConfig(
            output_dir=output_dir,
            learning_rate=self.learning_rate,  # Use converted float value
            per_device_train_batch_size=self.batch_size,  # Use converted int value
            beta=self.beta,  # Use converted float value
            logging_steps=1,  # Log every step for dense metrics
            save_strategy="no",  # Disable checkpoint saving during training (avoids tokenizer serialization issues)
            num_train_epochs=1,
            remove_unused_columns=False,
            gradient_checkpointing=self.gradient_checkpointing,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
            max_length=self.max_length,  # Limit sequence length to reduce memory
            report_to=[],  # Disable wandb/TensorBoard
            logging_first_step=True,  # Log the first step
            run_name=wandb_run_name if wandb_run_name else None,  # Optional experiment label
            bf16=self.use_bf16,
            fp16=self.use_fp16,
            max_grad_norm=1.0,
        )
        
        # DPOTrainer will be initialized when we have training data
        self.dpo_trainer = None
        self.peft_config = peft_config
        self.global_step = 0  # Track global step across iterations
        print(f"[DPO Trainer] Initialization complete!\n")
        
        self.training_history = []

        # Load fixed reference model to keep KL term anchored
        self.ref_model = self._load_reference_model()

    def _ensure_reference_model_device(self):
        """Keep frozen reference on the same device as policy model."""
        if self.ref_model is None:
            return
        try:
            ref_device = next(self.ref_model.parameters()).device
            desired_device = torch.device(self.ref_device) if self.ref_device else self.device
            if ref_device != desired_device:
                self.ref_model.to(desired_device)
        except Exception as e:
            print(f"  ⚠️ Failed to place reference model on policy device: {e}")

    def _prepare_policy_trainable_params(self):
        """Enable gradients for intended parameters and return the trainable list."""
        if self.use_lora:
            for name, param in self.model.named_parameters():
                lower_name = name.lower()
                is_adapter = ("lora" in lower_name) or ("adapter" in lower_name)
                param.requires_grad_(is_adapter)
        else:
            for _, param in self.model.named_parameters():
                param.requires_grad_(True)
        return [p for p in self.model.parameters() if p.requires_grad]

    def _completion_logprob(self, model, prompt: str, completion: str, reduce: str = "mean"):
        """
        Compute log-probability of `completion` tokens conditioned on chat-style `prompt`.
        """
        model_device = next(model.parameters()).device
        if not completion:
            return torch.tensor(0.0, device=model_device)

        messages = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(model_device)

        completion_ids = self.tokenizer(
            completion,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(model_device)

        if completion_ids.shape[1] == 0:
            return torch.tensor(0.0, device=model_device)

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # Bound sequence length to control activation memory in RL updates.
        if full_ids.shape[1] > self.max_length:
            full_ids = full_ids[:, -self.max_length :]

        attention_mask = torch.ones_like(full_ids)
        outputs = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)

        logits = outputs.logits[:, :-1, :]
        targets = full_ids[:, 1:]
        # Completion tokens are at the tail of sequence after concatenation.
        completion_len = min(completion_ids.shape[1], targets.shape[1])
        if completion_len <= 0:
            return torch.tensor(0.0, device=model_device)
        completion_logits = logits[:, -completion_len:, :]
        completion_targets = targets[:, -completion_len:]
        token_log_probs = F.log_softmax(completion_logits, dim=-1).gather(
            -1, completion_targets.unsqueeze(-1)
        ).squeeze(-1)

        if reduce == "sum":
            return token_log_probs.sum()
        if reduce == "none":
            return token_log_probs
        return token_log_probs.mean()

    def train_rule_batch(
        self,
        samples: List[RuleRLSample],
        num_epochs: int = 1,
        kl_coef: float = 0.02,
        normalize_advantage: bool = True,
        max_grad_norm: float = 1.0,
    ) -> Dict:
        """
        KL-regularized policy-gradient update:
          loss = -A * log pi(y|x) + kl_coef * (log pi(y|x) - log pref(y|x))
        """
        if len(samples) == 0:
            print("  No rule samples available for RL training")
            return {"num_samples": 0}

        self.device = next(self.model.parameters()).device
        self._ensure_reference_model_device()
        self.model.train()
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        trainable_params = self._prepare_policy_trainable_params()
        if not trainable_params:
            print("  ⚠️ No trainable parameters found for policy-gradient update")
            return {"num_samples": len(samples), "skipped": "no_trainable_params"}

        rewards = torch.tensor(
            [float(sample.reward) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = rewards - rewards.mean()
        if normalize_advantage and len(samples) > 1:
            advantages = advantages / (rewards.std(unbiased=False) + 1e-8)

        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        epoch_losses = []
        epoch_kl = []
        epoch_logp = []

        print(
            f"  RL policy update on {len(samples)} samples for {num_epochs} epoch(s); "
            f"reward range=[{rewards.min().item():.4f}, {rewards.max().item():.4f}], kl_coef={kl_coef}"
        )

        epoch_iterator = tqdm(
            range(num_epochs),
            desc="RL epochs",
            leave=False,
        )
        for epoch in epoch_iterator:
            optimizer.zero_grad()
            grad_accum = max(1, int(self.gradient_accumulation_steps))
            num_samples = len(samples)
            per_sample_losses = []
            per_sample_kl = []
            per_sample_logp = []

            sample_iterator = tqdm(
                enumerate(samples),
                total=num_samples,
                desc=f"RL step {epoch + 1}/{num_epochs}",
                leave=False,
            )
            for idx, sample in sample_iterator:
                logp_policy = self._completion_logprob(
                    self.model, sample.prompt, sample.completion, reduce="mean"
                )
                if self.ref_model is not None:
                    with torch.no_grad():
                        logp_ref = self._completion_logprob(
                            self.ref_model, sample.prompt, sample.completion, reduce="mean"
                        )
                    logp_ref = logp_ref.to(logp_policy.device)
                else:
                    logp_ref = torch.zeros_like(logp_policy)

                kl_estimate = logp_policy - logp_ref
                pg_term = -(advantages[idx].detach() * logp_policy)
                loss = pg_term + (kl_coef * kl_estimate)
                scaled_loss = loss / grad_accum
                scaled_loss.backward()
                per_sample_losses.append(loss.detach())
                per_sample_kl.append(kl_estimate.detach())
                per_sample_logp.append(logp_policy.detach())

                step_boundary = ((idx + 1) % grad_accum == 0) or (idx + 1 == num_samples)
                if step_boundary:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                sample_iterator.set_postfix(
                    loss=float(loss.detach().cpu().item()),
                    kl=float(kl_estimate.detach().cpu().item()),
                    refresh=False,
                )

            batch_loss = torch.stack(per_sample_losses).mean()

            epoch_losses.append(float(batch_loss.detach().cpu().item()))
            epoch_kl.append(float(torch.stack(per_sample_kl).mean().cpu().item()))
            epoch_logp.append(float(torch.stack(per_sample_logp).mean().cpu().item()))
            epoch_iterator.set_postfix(
                loss=epoch_losses[-1],
                kl=epoch_kl[-1],
                logp=epoch_logp[-1],
                refresh=False,
            )
            print(
                f"    Epoch {epoch + 1}/{num_epochs}: "
                f"loss={epoch_losses[-1]:.6f}, kl={epoch_kl[-1]:.6f}, logp={epoch_logp[-1]:.6f}"
            )

        metrics = {
            "num_samples": len(samples),
            "num_epochs": num_epochs,
            "mean_reward": float(rewards.mean().detach().cpu().item()),
            "std_reward": float(rewards.std(unbiased=False).detach().cpu().item()),
            "rl_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "rl_kl": float(epoch_kl[-1]) if epoch_kl else 0.0,
            "mean_logp": float(epoch_logp[-1]) if epoch_logp else 0.0,
            "epoch_losses": epoch_losses,
        }
        self.training_history.append(metrics)
        return metrics
    
    def _load_reference_model(self):
        """
        Load a frozen reference model used for the DPO KL term.
        The reference must stay fixed across batches/iterations.
        """
        if self.ref_model_path is None:
            print("  ⚠️ No ref_model_path provided; reference model will drift (not recommended).")
            return None
        try:
            ref_source = self.ref_model_path
            local_exists = os.path.exists(ref_source)
            if local_exists:
                ref_source = os.path.abspath(ref_source)
            print(f"  Loading frozen reference model from {ref_source}...")
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_source,
                torch_dtype=self.model.dtype,
                device_map=None,  # place on single device explicitly below
                token=self.hf_token,
                cache_dir=self.hf_cache_dir,
                local_files_only=local_exists,
            )
            # Place reference on configured device (defaults to policy device if unset).
            target_device = torch.device(self.ref_device) if self.ref_device else self.device
            ref_model.to(target_device)
            print(f"  Reference model device: {target_device}")
            for p in ref_model.parameters():
                p.requires_grad_(False)
            ref_model.eval()
            return ref_model
        except Exception as e:
            print(f"  ⚠️ Failed to load reference model from {self.ref_model_path}: {e}")
            return None
    
    class GlobalStepCallback(TrainerCallback):
        """Callback to set initial global step for continuous logging"""
        def __init__(self, initial_step: int = 0):
            self.initial_step = initial_step
            self.set = False
        
        def on_train_begin(self, args, state, control, **kwargs):
            """Set global step at the start of training"""
            if not self.set and self.initial_step > 0:
                state.global_step = self.initial_step
                self.set = True
    
    def _create_dpo_dataset(self, training_pairs: List[DPOTrainingPair]) -> Dataset:
        """
        Convert DPO training pairs to HuggingFace Dataset format
        
        DPO format requires:
        - prompt: Input context (system + user prompt)
        - chosen: Preferred response (the generated prompt with high reward)
        - rejected: Less preferred response (the generated prompt with low reward)
        - score_chosen: Reward for chosen (optional, for logging)
        - score_rejected: Reward for rejected (optional, for logging)
        
        Args:
            training_pairs: List of DPOTrainingPair objects
        
        Returns:
            HuggingFace Dataset with DPO format
        """
        # Format data for DPO: prompt, chosen, rejected, score_chosen, score_rejected
        dataset_dict = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
            "score_chosen": [],
            "score_rejected": []
        }
        
        for pair in training_pairs:
            dataset_dict["prompt"].append(pair.prompt)
            # For conversational models, chosen/rejected should be the assistant's response
            # (the generated prompt text)
            dataset_dict["chosen"].append(pair.chosen)
            dataset_dict["rejected"].append(pair.rejected)
            dataset_dict["score_chosen"].append(pair.score_chosen)
            dataset_dict["score_rejected"].append(pair.score_rejected)
        
        return Dataset.from_dict(dataset_dict)
    
    def train_batch(
        self,
        training_pairs: List[DPOTrainingPair],
        num_epochs: int = 1
    ) -> Dict:
        """
        Train on a batch of DPO preference pairs
        
        Args:
            training_pairs: List of DPOTrainingPair objects (chosen vs rejected)
            num_epochs: Number of training epochs
        
        Returns:
            Dictionary with training metrics
        """
        if len(training_pairs) == 0:
            print("  No training pairs available")
            return {"num_samples": 0}
        
        print(f"  Training on {len(training_pairs)} DPO pairs for {num_epochs} epoch(s)...")
        print(f"  Chosen reward range: [{min(p.score_chosen for p in training_pairs):.3f}, {max(p.score_chosen for p in training_pairs):.3f}]")
        print(f"  Rejected reward range: [{min(p.score_rejected for p in training_pairs):.3f}, {max(p.score_rejected for p in training_pairs):.3f}]")
        print(
            f"  DPO settings: per_device_batch={self.dpo_config.per_device_train_batch_size}, "
            f"grad_accum={self.dpo_config.gradient_accumulation_steps}, "
            f"max_length={self.dpo_config.max_length}, "
            f"epochs={num_epochs}, "
            f"bf16={self.dpo_config.bf16}, fp16={self.dpo_config.fp16}"
        )
        
        # Create dataset
        train_dataset = self._create_dpo_dataset(training_pairs)
        
        # Update DPO config with number of epochs and keep logging frequent
        self.dpo_config.num_train_epochs = num_epochs
        self.dpo_config.logging_steps = 1  # Keep frequent logs for loss curve
        self.dpo_config.logging_first_step = True
        # Ensure run name is set (optional metadata only)
        if self.wandb_run_name:
            self.dpo_config.run_name = self.wandb_run_name
        # Allow beta override/anneal per batch in the future by mutating self.dpo_config.beta before trainer init
        
        # Recreate DPOTrainer for each training batch to ensure fresh state and logging
        # This ensures a clean trainer state for each iteration
        print(f"  Initializing DPOTrainer...")
        # Configure gradient checkpointing based on config
        if self.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print(f"  ✓ Gradient checkpointing enabled")
        elif not self.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_disable'):
            try:
                self.model.gradient_checkpointing_disable()
                print(f"  ✓ Gradient checkpointing disabled")
            except AttributeError:
                # Some model versions may not have installed the grad hook; safe to ignore
                print(f"  ⚠️ Gradient checkpointing disable skipped (hook not set)")
        # Ensure model is in train mode
        self.model.train()
        # If using LoRA, explicitly mark adapter params trainable (some checkpoints load frozen)
        if self.use_lora:
            for n, p in self.model.named_parameters():
                if "lora" in n or "adapter" in n:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        # Log trainable parameter count to detect frozen model issues
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable_params:,} / {total_params:,}")
        
        # Create callback to set initial global step for continuous logging
        callbacks = []
        if self.global_step > 0:
            step_callback = self.GlobalStepCallback(initial_step=self.global_step)
            callbacks.append(step_callback)
            print(f"  Will resume from global step: {self.global_step}")
        
        # If the model already has PEFT adapters attached, avoid passing a new peft_config
        peft_config_to_use = self.peft_config
        if self.peft_config is not None:
            model_has_adapters = hasattr(self.model, "peft_config")
            try:
                from peft import PeftModel
                model_has_adapters = model_has_adapters or isinstance(self.model, PeftModel)
            except Exception:
                pass
            if model_has_adapters:
                peft_config_to_use = None
                print("  Model already has PEFT adapters; skipping peft_config to avoid duplicate LoRA wrapping")
        
        # Force single-GPU in Trainer even if multiple devices are visible (avoid torch.nn.DataParallel hang)
        self.dpo_config._n_gpu = 1
        if hasattr(self.dpo_config, "local_rank"):
            self.dpo_config.local_rank = -1
        original_device_count = torch.cuda.device_count
        torch.cuda.device_count = lambda: 1  # type: ignore
        try:
            self.dpo_trainer = DPOTrainer(
                model=self.model,
                args=self.dpo_config,
                processing_class=self.tokenizer,
                train_dataset=train_dataset,
                peft_config=peft_config_to_use,
                ref_model=self.ref_model,  # Fixed, frozen reference model to anchor KL term
                callbacks=callbacks if callbacks else None,  # Add callback to set initial step
            )
        finally:
            torch.cuda.device_count = original_device_count  # type: ignore
        # Explicitly tell Trainer not to data-parallelize
        self.dpo_trainer.n_gpu = 1
        if hasattr(self.dpo_trainer, "args"):
            self.dpo_trainer.args._n_gpu = 1
            try:
                # Some HF versions make parallel_mode a property; bypass by setting protected member
                if hasattr(self.dpo_trainer.args, "_parallel_mode"):
                    self.dpo_trainer.args._parallel_mode = ParallelMode.NOT_DISTRIBUTED
                elif hasattr(self.dpo_trainer.args, "parallel_mode"):
                    object.__setattr__(self.dpo_trainer.args, "parallel_mode", ParallelMode.NOT_DISTRIBUTED)
            except Exception as e:
                print(f"  Warning: could not force parallel_mode to NOT_DISTRIBUTED: {e}")
        try:
            n_gpu = self.dpo_trainer.args.n_gpu if hasattr(self.dpo_trainer, "args") else "?"
            pmode = getattr(self.dpo_trainer.args, "parallel_mode", "?") if hasattr(self.dpo_trainer, "args") else "?"
            print(f"  Trainer device setup: n_gpu={n_gpu}, parallel_mode={pmode}, cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
        except Exception as e:
            print(f"  Trainer device setup logging failed: {e}")
        print(f"  ✓ DPOTrainer initialized")
        
        # Train
        print(f"  Starting DPO training...")
        train_result = self.dpo_trainer.train()
        
        # Update global step counter for next iteration
        if hasattr(self.dpo_trainer, 'state'):
            self.global_step = self.dpo_trainer.state.global_step
            print(f"  Training completed at global step: {self.global_step}")
        
        # Extract metrics
        training_metrics = {
            "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else 0.0,
            "num_samples": len(training_pairs),
            "num_epochs": num_epochs,
            "mean_chosen_reward": float(np.mean([p.score_chosen for p in training_pairs])),
            "mean_rejected_reward": float(np.mean([p.score_rejected for p in training_pairs])),
            "global_step": self.global_step,
        }
        
        # Store training history
        self.training_history.append(training_metrics)
        
        print(f"  ✓ Training complete!")
        print(f"    Training loss: {training_metrics['train_loss']:.4f}")
        print(f"    Mean chosen reward: {training_metrics['mean_chosen_reward']:.3f}")
        print(f"    Mean rejected reward: {training_metrics['mean_rejected_reward']:.3f}")
        # Extract and plot per-epoch losses
        epoch_losses = self._extract_epoch_losses()
        if epoch_losses:
            training_metrics["epoch_losses"] = epoch_losses
            self._save_epoch_losses_csv(epoch_losses)
            self._plot_epoch_losses(epoch_losses)
        return training_metrics
    
    @staticmethod
    def _model_has_invalid_weights(model) -> bool:
        """Check if model parameters contain NaN or Inf values."""
        import torch
        for p in model.parameters():
            if p is not None:
                if torch.isnan(p).any() or torch.isinf(p).any():
                    return True
        return False

    def _extract_epoch_losses(self) -> List[Dict[str, float]]:
        """
        Pull per-epoch loss from trainer log history.
        Returns list of dicts with keys epoch and loss.
        """
        if self.dpo_trainer is None or not hasattr(self.dpo_trainer, "state"):
            return []
        history = getattr(self.dpo_trainer.state, "log_history", None) or []
        latest_per_epoch: Dict[float, float] = {}
        for entry in history:
            epoch = entry.get("epoch")
            if epoch is None:
                continue
            # Prefer train_loss if present, else loss
            loss_val = entry.get("train_loss", entry.get("loss"))
            if loss_val is None:
                continue
            latest_per_epoch[float(epoch)] = float(loss_val)
        if not latest_per_epoch:
            return []
        return [{"epoch": e, "loss": latest_per_epoch[e]} for e in sorted(latest_per_epoch.keys())]

    def _plot_epoch_losses(self, epoch_losses: List[Dict[str, float]]):
        """Plot loss per epoch and save to output_dir/dpo_epoch_loss.png"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  ⚠️ matplotlib not available; skipping loss plot (CSV still saved).")
            return
        epochs = [item["epoch"] for item in epoch_losses]
        losses = [item["loss"] for item in epoch_losses]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("DPO Training Loss per Epoch")
        plt.grid(True, linestyle="--", alpha=0.6)
        os.makedirs(self.output_dir, exist_ok=True)
        plot_path = os.path.join(self.output_dir, "dpo_epoch_loss.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved loss plot to {plot_path}")

    def _save_epoch_losses_csv(self, epoch_losses: List[Dict[str, float]]):
        """Persist epoch/loss pairs so they can be plotted elsewhere if matplotlib is unavailable."""
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "dpo_epoch_loss.csv")
        with open(csv_path, "w") as f:
            f.write("epoch,loss\n")
            for item in epoch_losses:
                f.write(f"{item['epoch']},{item['loss']}\n")
        print(f"  ✓ Saved epoch loss table to {csv_path}")
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model"""
        # Save model (works for both regular model and PEFT/LoRA models)
        if self.dpo_trainer is not None and hasattr(self.dpo_trainer, 'model'):
            # Use the model from DPO trainer (may have LoRA adapters)
            model_to_save = self.dpo_trainer.model
        else:
            model_to_save = self.model

        # Skip saving if weights are invalid
        if self._model_has_invalid_weights(model_to_save):
            print(f"  ⚠️ Detected NaN/Inf in model weights; skipping save to avoid corrupt checkpoints.")
            return

        save_path = path or self.output_dir
        os.makedirs(save_path, exist_ok=True)
        
        print(f"  Saving model to {save_path}...")
        
        # Save model (this is the most important part)
        model_to_save.save_pretrained(save_path)
        # Save tokenizer alongside for easier reload
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)
        print(f"  ✓ Model saved successfully to {save_path}")
