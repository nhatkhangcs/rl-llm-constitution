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
)
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
        use_lora,
        lora_r,
        lora_alpha,
        output_dir: str = "./models/whitebox_dpo",
        beta: float = 0.1  # DPO beta parameter (temperature)
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
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.output_dir = output_dir
        self.beta = float(beta) if not isinstance(beta, float) else beta
        
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
            logging_steps=10,
            save_steps=100,
            num_train_epochs=1,
            remove_unused_columns=False,
        )
        
        # DPOTrainer will be initialized when we have training data
        self.dpo_trainer = None
        self.peft_config = peft_config
        print(f"[DPO Trainer] Initialization complete!\n")
        
        self.training_history = []
    
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
        
        # Create dataset
        train_dataset = self._create_dpo_dataset(training_pairs)
        
        # Update DPO config with number of epochs
        self.dpo_config.num_train_epochs = num_epochs
        
        # Initialize DPOTrainer if not already initialized
        if self.dpo_trainer is None:
            print(f"  Initializing DPOTrainer...")
            self.dpo_trainer = DPOTrainer(
                model=self.model,
                args=self.dpo_config,
                processing_class=self.tokenizer,
                train_dataset=train_dataset,
                peft_config=self.peft_config,
            )
            print(f"  ✓ DPOTrainer initialized")
        else:
            # Update dataset
            self.dpo_trainer.train_dataset = train_dataset
        
        # Train
        print(f"  Starting DPO training...")
        train_result = self.dpo_trainer.train()
        
        # Extract metrics
        training_metrics = {
            "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else 0.0,
            "num_samples": len(training_pairs),
            "num_epochs": num_epochs,
            "mean_chosen_reward": float(np.mean([p.score_chosen for p in training_pairs])),
            "mean_rejected_reward": float(np.mean([p.score_rejected for p in training_pairs])),
        }
        
        # Store training history
        self.training_history.append(training_metrics)
        
        print(f"  ✓ Training complete!")
        print(f"    Training loss: {training_metrics['train_loss']:.4f}")
        print(f"    Mean chosen reward: {training_metrics['mean_chosen_reward']:.3f}")
        print(f"    Mean rejected reward: {training_metrics['mean_rejected_reward']:.3f}")
        
        return training_metrics
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model"""
        save_path = path or self.output_dir
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Saving model to {save_path}")
        if self.dpo_trainer is not None:
            self.dpo_trainer.save_model(save_path)
        else:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        
        print(f"Model saved successfully!")

