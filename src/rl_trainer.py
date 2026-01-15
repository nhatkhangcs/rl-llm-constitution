"""
Reinforcement Learning Trainer for White-box LLM

This module implements DPO (Direct Preference Optimization) training for the white-box LLM using TRL library.
The model is trained to generate better prompts based on preference pairs (chosen vs rejected).
"""

import os
import torch
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

    



class RLTrainer:
    """Trains white-box LLM using DPO (Direct Preference Optimization)"""
    
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
        wandb_run_name: Optional[str] = None  # Wandb run name (for reusing same run across iterations)
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
            logging_steps=1,  # Log every step to ensure wandb updates frequently
            save_strategy="no",  # Disable checkpoint saving during training (avoids tokenizer serialization issues)
            num_train_epochs=1,
            remove_unused_columns=False,
            gradient_checkpointing=self.gradient_checkpointing,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
            max_length=self.max_length,  # Limit sequence length to reduce memory
            report_to="wandb",  # Explicitly enable wandb reporting
            logging_first_step=True,  # Log the first step
            run_name=wandb_run_name if wandb_run_name else None,  # Use same run name across iterations
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
    
    class GlobalStepCallback(TrainerCallback):
        """Callback to set initial global step for continuous wandb logging"""
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
        
        # Update DPO config with number of epochs and ensure logging is enabled
        self.dpo_config.num_train_epochs = num_epochs
        self.dpo_config.logging_steps = 1  # Log every step to ensure wandb updates frequently
        self.dpo_config.logging_first_step = True
        # Ensure wandb run name is set (for reusing same run across iterations)
        if self.wandb_run_name:
            self.dpo_config.run_name = self.wandb_run_name
        
        # Recreate DPOTrainer for each training batch to ensure proper wandb logging
        # This ensures a fresh trainer state and proper logging for each iteration
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
        
        # Create callback to set initial global step for continuous wandb logging
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
                ref_model=None,  # Use None to share reference model (saves memory)
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
