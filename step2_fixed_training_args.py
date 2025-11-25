#!/usr/bin/env python3
"""
Memory-optimized OpenVLA fine-tuning script
FIXED: 8-bit quantization device placement issue
"""

import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenVLADataset(Dataset):
    """Dataset for OpenVLA fine-tuning - FIXED VERSION."""
    
    def __init__(self, data_dir, processor, max_length=128):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        
        # Load dataset summary
        summary_path = os.path.join(data_dir, "dataset_summary.json")
        with open(summary_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        self.samples = self.dataset_info["samples"]
        print(f"📊 Loaded dataset with {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image and resize to reduce memory
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Resize to save memory
        
        # Get instruction and action
        instruction = sample["instruction"]
        action = np.array(sample["action"])
        
        # Format action as text
        x, y, z, qx, qy, qz, qw = action
        action_text = f"move_to_pose({x:.3f}, {y:.3f}, {z:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})"
        
        # Create FULL text (instruction + action)
        full_text = f"Task: {instruction} Action: {action_text}"
        
        # Process with the FULL text
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Labels are the same as input_ids (for causal LM)
        labels = inputs["input_ids"].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Return only the necessary fields
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.squeeze(0)
        }


class OpenVLATrainer:
    """Trainer for OpenVLA fine-tuning with LoRA - MEMORY OPTIMIZED."""
    
    def __init__(self, model_name="openvla/openvla-7b", output_dir="./openvla_finetuned"):
        self.model_name = model_name
        self.output_dir = output_dir
        # Don't store device for quantized models - let them handle device placement automatically
        
        print(f"🚀 Initializing OpenVLA trainer...")
        print(f"📦 Model: {model_name}")
        print(f"💾 Output directory: {output_dir}")
        print(f"🔧 Using automatic device placement for quantized model")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_model_and_processor(self):
        """Load OpenVLA model with 8-bit quantization - FIXED VERSION."""
        print("📥 Loading OpenVLA model with 8-bit quantization...")
        
        # Set longer timeout for downloads
        import os
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
        
        try:
            # Load processor first
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Configure 8-bit quantization properly
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=False,
            )
            
            # Load model with proper quantization config
            # CRITICAL: Don't call .to() on quantized models!
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,  # Use proper config instead of load_in_8bit=True
                trust_remote_code=True,
                device_map="auto",  # Let it handle device placement automatically
                torch_dtype=torch.float16,  # Use float16 for efficiency
                attn_implementation="eager"
            )
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            print("✅ Model loaded successfully with 8-bit quantization!")
            print(f"🔧 Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            print("\n⚠️ Make sure you have installed:")
            print("pip install bitsandbytes accelerate")
            print("pip install --upgrade transformers")
            raise e
    
    def setup_lora(self, r=8, alpha=16, dropout=0.1):
        """Setup LoRA with reduced rank to save memory."""
        print(f"🔧 Setting up LoRA (r={r}, alpha={alpha}, dropout={dropout})...")
        
        # LoRA configuration with reduced rank
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,  # Reduced rank (8 instead of 16)
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],  # Only target q and v projections
            bias="none",
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        print("✅ LoRA setup complete!")
    
    def prepare_training_arguments(self, num_epochs=5, batch_size=1, learning_rate=2e-4):
        """Prepare memory-optimized training arguments."""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  # Effective batch size = 8
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=1,
            save_steps=50,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=25,
            warmup_steps=5,
            fp16=True,  # Enable fp16 for memory savings with quantized model
            gradient_checkpointing=True,  # Save memory
            dataloader_pin_memory=False,  # Save memory
            remove_unused_columns=False,
            report_to="none",  # Disable wandb
            load_best_model_at_end=False,  # Save memory
            optim="adamw_8bit",  # Use 8-bit optimizer
            max_grad_norm=0.3,  # Gradient clipping
            # Additional memory optimizations
            dataloader_num_workers=0,  # Reduce memory usage
            prediction_loss_only=True,  # Only compute loss, not all outputs
        )
        
        return training_args
    
    def train(self, dataset, training_args):
        """Train the model - FIXED VERSION."""
        print("🏋️ Starting training...")
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Split dataset (80% train, 20% eval)
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset, [train_size, eval_size]
        )
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Eval samples: {len(eval_dataset)}")
        
        # Custom data collator to handle the specific format
        def data_collator(features):
            batch = {}
            for key in features[0].keys():
                batch[key] = torch.stack([f[key] for f in features])
            return batch
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processor.tokenizer,
            data_collator=data_collator,  # Use custom data collator
        )
        
        # Train
        try:
            trainer.train()
        except Exception as e:
            print(f"❌ Training error: {e}")
            print("🔄 Trying with smaller batch size...")
            # Clear cache and retry
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
        
        # Save the final model
        print("💾 Saving final model...")
        trainer.save_model(os.path.join(self.output_dir, "final_model"))
        self.processor.save_pretrained(os.path.join(self.output_dir, "final_model"))
        
        print("✅ Training complete!")
        
        return trainer
    
    def save_training_info(self, dataset_info, training_args):
        """Save training information."""
        training_info = {
            "model_name": self.model_name,
            "training_date": datetime.now().isoformat(),
            "dataset_info": dataset_info,
            "quantization": "8-bit (BitsAndBytesConfig)",
            "training_args": {
                "num_train_epochs": training_args.num_train_epochs,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            },
            "lora_config": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            }
        }
        
        info_path = os.path.join(self.output_dir, "training_info.json")
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"📋 Training info saved to {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA with LoRA (Memory Optimized - FIXED)")
    parser.add_argument("--data_dir", type=str, default="./openvla_training_data", 
                       help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="./openvla_finetuned", 
                       help="Directory to save fine-tuned model")
    parser.add_argument("--model_name", type=str, default="openvla/openvla-7b", 
                       help="OpenVLA model name")
    parser.add_argument("--num_epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist. Run step1_data_collection.py first.")
    
    summary_path = os.path.join(args.data_dir, "dataset_summary.json")
    if not os.path.exists(summary_path):
        raise ValueError(f"Dataset summary not found at {summary_path}. Run step1_data_collection.py first.")
    
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🔧 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("🎯 OpenVLA Fine-tuning Pipeline (Memory Optimized - FIXED)")
    print("=" * 60)
    
    # Initialize trainer
    trainer = OpenVLATrainer(model_name=args.model_name, output_dir=args.output_dir)
    
    # Load model and processor
    trainer.load_model_and_processor()
    
    # Setup LoRA with reduced parameters
    trainer.setup_lora(r=8, alpha=16, dropout=0.1)
    
    # Create dataset
    print("📊 Creating dataset...")
    dataset = OpenVLADataset(args.data_dir, trainer.processor, max_length=128)
    
    # Prepare training arguments
    training_args = trainer.prepare_training_arguments(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save training info
    trainer.save_training_info(dataset.dataset_info, training_args)
    
    # Train model
    try:
        trained_model = trainer.train(dataset, training_args)
        print("🎉 Fine-tuning complete!")
        print(f"📁 Model saved to: {args.output_dir}")
        print("🔄 You can now use the fine-tuned model for inference")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Try reducing batch size or using a smaller model")
        raise e


if __name__ == "__main__":
    main()