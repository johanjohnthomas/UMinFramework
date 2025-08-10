"""
Fine-tuning script for T5 Prompt Refiner model.
Trains a T5-Small model on AskCQ dataset for prompt clarification.
"""
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Try to import transformers and related libraries
try:
    from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback
    )
    from datasets import Dataset, DatasetDict, load_from_disk
    import torch
    import numpy as np
    HAS_TRANSFORMERS = True
except ImportError as e:
    print(f"Warning: Could not import transformers/datasets: {e}")
    print("Please install the required packages: pip install transformers datasets torch")
    HAS_TRANSFORMERS = False
    # Create dummy classes for type hints
    DatasetDict = None
    TrainingArguments = None


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('finetune_prompt_refiner.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune T5 model for prompt refinement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing preprocessed training data')
    parser.add_argument('--output-dir', type=str, default='./models/prompt_refiner',
                       help='Directory to save the fine-tuned model')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='t5-small',
                       help='Base T5 model to fine-tune')
    parser.add_argument('--max-input-length', type=int, default=512,
                       help='Maximum input sequence length')
    parser.add_argument('--max-target-length', type=int, default=256,
                       help='Maximum target sequence length')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size per device')
    parser.add_argument('--eval-batch-size', type=int, default=16,
                       help='Evaluation batch size per device')
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--warmup-steps', type=int, default=500,
                       help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Evaluation and saving
    parser.add_argument('--eval-steps', type=int, default=100,
                       help='Number of steps between evaluations')
    parser.add_argument('--save-steps', type=int, default=500,
                       help='Number of steps between checkpoint saves')
    parser.add_argument('--save-total-limit', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                       help='Early stopping patience (number of evaluations)')
    
    # Hardware and optimization
    parser.add_argument('--fp16', action='store_true',
                       help='Use 16-bit mixed precision training')
    parser.add_argument('--dataloader-num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    # Logging and debugging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--logging-steps', type=int, default=50,
                       help='Number of steps between logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Execution mode
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without training (for testing setup)')
    parser.add_argument('--resume-from-checkpoint', type=str,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, logger):
    """Load T5 model and tokenizer."""
    if not HAS_TRANSFORMERS:
        logger.error("Transformers not available. Please install required packages.")
        return None, None
    
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        logger.info(f"✓ Loaded tokenizer: {model_name}")
        
        # Load model
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info(f"✓ Loaded model: {model_name}")
        logger.info(f"Model parameters: {model.num_parameters():,}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        return None, None


def load_preprocessed_data(data_dir: str, logger):
    """Load preprocessed training data."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return None
    
    logger.info(f"Loading preprocessed data from: {data_path}")
    
    try:
        if HAS_TRANSFORMERS:
            # Try to load Hugging Face dataset format first
            if (data_path / "dataset_dict.json").exists():
                dataset_dict = load_from_disk(str(data_path))
                logger.info(f"✓ Loaded HF dataset format")
            else:
                # Load from JSON files
                train_path = data_path / "train.json"
                val_path = data_path / "validation.json"
                
                if not train_path.exists() or not val_path.exists():
                    logger.error(f"Required files not found: {train_path}, {val_path}")
                    return None
                
                with open(train_path, 'r') as f:
                    train_data = json.load(f)
                
                with open(val_path, 'r') as f:
                    val_data = json.load(f)
                
                dataset_dict = DatasetDict({
                    'train': Dataset.from_list(train_data),
                    'validation': Dataset.from_list(val_data)
                })
                logger.info(f"✓ Loaded JSON format")
        else:
            logger.error("Transformers not available - cannot load datasets")
            return None
        
        logger.info(f"Training examples: {len(dataset_dict['train'])}")
        logger.info(f"Validation examples: {len(dataset_dict['validation'])}")
        
        return dataset_dict
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None


def prepare_training_arguments(args, output_dir: str):
    """Prepare training arguments for Hugging Face Trainer."""
    if not HAS_TRANSFORMERS:
        return None
    
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to=[],  # Disable wandb/tensorboard for simplicity
        
        # Hardware optimization
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        
        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
        
        # Other
        remove_unused_columns=False,
        push_to_hub=False,
    )


def create_trainer(model, tokenizer, dataset_dict, training_args, logger):
    """Create and configure the Trainer."""
    if not HAS_TRANSFORMERS:
        return None
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    logger.info("✓ Trainer configured successfully")
    return trainer


def save_model_artifacts(model, tokenizer, output_dir: str, logger):
    """Save model, tokenizer, and training configuration."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save model and tokenizer
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"✓ Model and tokenizer saved to {output_path}")
        
        # Save model configuration
        config_file = output_path / "training_config.json"
        config = {
            "base_model": "t5-small",
            "task": "prompt_refinement", 
            "training_framework": "huggingface_transformers"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Training config saved to {config_file}")
        
    except Exception as e:
        logger.error(f"Failed to save model artifacts: {e}")


def main():
    """Main fine-tuning function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting T5 Prompt Refiner fine-tuning")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check if transformers is available
    if not HAS_TRANSFORMERS:
        logger.error("Transformers library not available. Exiting.")
        return
    
    # Set random seed
    if HAS_TRANSFORMERS:
        from transformers import set_seed
        set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, logger)
    if model is None or tokenizer is None:
        logger.error("Failed to load model and tokenizer. Exiting.")
        return
    
    # Load preprocessed data
    dataset_dict = load_preprocessed_data(args.data_dir, logger)
    if dataset_dict is None:
        logger.error("Failed to load training data. Exiting.")
        return
    
    # Check if we have enough data
    if len(dataset_dict['train']) == 0:
        logger.warning("No training data found. Creating placeholder for testing.")
        if args.dry_run:
            logger.info("✓ Dry run completed successfully")
            return
        else:
            logger.error("Cannot train with no data. Exiting.")
            return
    
    if args.dry_run:
        logger.info("✓ Dry run completed successfully - all components loaded correctly")
        return
    
    # Prepare training arguments
    training_args = prepare_training_arguments(args, str(output_dir))
    if training_args is None:
        logger.error("Failed to prepare training arguments. Exiting.")
        return
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset_dict, training_args, logger)
    if trainer is None:
        logger.error("Failed to create trainer. Exiting.")
        return
    
    logger.info("Starting fine-tuning...")
    logger.info(f"Training samples: {len(dataset_dict['train'])}")
    logger.info(f"Validation samples: {len(dataset_dict['validation'])}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Start training
    try:
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
        
        logger.info("✓ Training completed successfully")
        
        # Save the final model
        save_model_artifacts(model, tokenizer, str(output_dir), logger)
        
        logger.info(f"✓ Fine-tuning pipeline completed. Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()