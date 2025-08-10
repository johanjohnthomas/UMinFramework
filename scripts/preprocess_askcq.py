"""
Data loading and preprocessing script for AskCQ dataset.
Formats the data for T5 model fine-tuning with prompt clarification task.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Try to import transformers/datasets, but make it optional for basic functionality
try:
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError as e:
    print(f"Warning: Could not import transformers/datasets: {e}")
    print("Running in basic mode without tokenization...")
    HAS_TRANSFORMERS = False


def load_askcq_data(data_path: str) -> List[Dict[str, Any]]:
    """Load AskCQ data from JSONL file."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"AskCQ data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data.append(item)
    
    print(f"✓ Loaded {len(data)} examples from {data_path}")
    return data


def create_training_pairs(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert AskCQ data into input-target pairs for T5 training.
    
    Format:
    - Input: "clarify: <ambiguous_question>"
    - Target: "<clarified_question>" or "<clarifying_question>"
    """
    training_pairs = []
    
    for item in data:
        question = item.get('question', '')
        annotations = item.get('annotations', [])
        
        if not question:
            continue
            
        # Create training pairs from annotations
        for annotation in annotations:
            if annotation.get('type') == 'ambiguous':
                clarification = annotation.get('answer', '')
                if clarification:
                    training_pairs.append({
                        'input_text': f"clarify: {question}",
                        'target_text': clarification
                    })
        
        # If no annotations, create a fallback clarification
        if not annotations:
            # Generate a generic clarification request
            training_pairs.append({
                'input_text': f"clarify: {question}",
                'target_text': f"Could you please be more specific about: {question.lower()}?"
            })
    
    print(f"✓ Created {len(training_pairs)} training pairs")
    return training_pairs


def augment_data(training_pairs: List[Dict[str, str]], augment_factor: int = 3) -> List[Dict[str, str]]:
    """
    Augment the training data with variations to improve model robustness.
    """
    augmented_pairs = training_pairs.copy()
    
    # Augmentation templates for clarification
    templates = [
        "Can you clarify what you mean by: {question}?",
        "I need more details about: {question}",
        "Could you be more specific about: {question}?",
        "What aspect of {question} are you asking about?",
        "Can you provide more context for: {question}?"
    ]
    
    for pair in training_pairs[:len(training_pairs)//augment_factor]:  # Only augment some examples
        question = pair['input_text'].replace('clarify: ', '')
        
        for i, template in enumerate(templates):
            if len(augmented_pairs) >= len(training_pairs) * augment_factor:
                break
                
            augmented_pairs.append({
                'input_text': f"clarify: {question}",
                'target_text': template.format(question=question.lower())
            })
    
    print(f"✓ Augmented dataset to {len(augmented_pairs)} examples")
    return augmented_pairs


def tokenize_data(data: List[Dict[str, str]], tokenizer, max_input_length: int = 512, 
                 max_target_length: int = 256):
    """
    Tokenize the input-target pairs for T5 training.
    """
    if not HAS_TRANSFORMERS:
        print("⚠ Tokenization skipped - transformers not available")
        return data
    
    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples['input_text'],
            max_length=max_input_length,
            truncation=True,
            padding=False
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target_text'],
                max_length=max_target_length,
                truncation=True,
                padding=False
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(data)
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"✓ Tokenized dataset with {len(tokenized_dataset)} examples")
    return tokenized_dataset


def create_train_val_split(dataset, val_ratio: float = 0.2):
    """Split dataset into train and validation sets."""
    if not HAS_TRANSFORMERS:
        # Simple split for basic data
        split_idx = int(len(dataset) * (1 - val_ratio))
        return {
            'train': dataset[:split_idx],
            'validation': dataset[split_idx:]
        }
    
    split_dataset = dataset.train_test_split(test_size=val_ratio, seed=42)
    
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })
    
    print(f"✓ Split dataset: {len(dataset_dict['train'])} train, {len(dataset_dict['validation'])} validation")
    return dataset_dict


def save_processed_data(dataset_dict, output_dir: str):
    """Save the processed dataset to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if HAS_TRANSFORMERS and hasattr(dataset_dict, 'save_to_disk'):
        # Save as Hugging Face dataset
        dataset_dict.save_to_disk(str(output_path))
        
        # Also save some examples for inspection
        examples_path = output_path / 'sample_examples.json'
        sample_data = []
        
        for i in range(min(5, len(dataset_dict['train']))):
            example = dataset_dict['train'][i]
            sample_data.append({
                'input_ids': example['input_ids'][:20],  # First 20 tokens
                'labels': example['labels'][:20],  # First 20 tokens
                'attention_mask': example['attention_mask'][:20]
            })
        
        with open(examples_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
            
        print(f"✓ Saved processed dataset to {output_path}")
        print(f"✓ Sample examples saved to {examples_path}")
    else:
        # Save as simple JSON files
        train_path = output_path / 'train.json'
        val_path = output_path / 'validation.json'
        
        with open(train_path, 'w') as f:
            json.dump(dataset_dict['train'], f, indent=2)
            
        with open(val_path, 'w') as f:
            json.dump(dataset_dict['validation'], f, indent=2)
            
        # Save sample examples
        examples_path = output_path / 'sample_examples.json'
        sample_data = dataset_dict['train'][:5] if len(dataset_dict['train']) >= 5 else dataset_dict['train']
        
        with open(examples_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
            
        print(f"✓ Saved processed dataset to {output_path}")
        print(f"✓ Training data: {train_path} ({len(dataset_dict['train'])} examples)")
        print(f"✓ Validation data: {val_path} ({len(dataset_dict['validation'])} examples)")
        print(f"✓ Sample examples saved to {examples_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess AskCQ data for T5 fine-tuning')
    parser.add_argument('--data-path', type=str, default='data/askcq.jsonl',
                       help='Path to AskCQ JSONL file')
    parser.add_argument('--output-dir', type=str, default='data/processed_askcq',
                       help='Output directory for processed data')
    parser.add_argument('--model-name', type=str, default='t5-small',
                       help='Model name for tokenizer')
    parser.add_argument('--max-input-length', type=int, default=512,
                       help='Maximum input sequence length')
    parser.add_argument('--max-target-length', type=int, default=256,
                       help='Maximum target sequence length')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--augment-factor', type=int, default=3,
                       help='Data augmentation factor')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AskCQ Data Preprocessing for T5 Fine-tuning")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading AskCQ data...")
    raw_data = load_askcq_data(args.data_path)
    
    # Create training pairs
    print("\n2. Creating input-target pairs...")
    training_pairs = create_training_pairs(raw_data)
    
    if len(training_pairs) == 0:
        print("❌ No training pairs created. Check your data format.")
        return
    
    # Augment data (optional)
    if not args.no_augment:
        print("\n3. Augmenting data...")
        training_pairs = augment_data(training_pairs, args.augment_factor)
    else:
        print("\n3. Skipping data augmentation...")
    
    # Load tokenizer
    print("\n4. Loading tokenizer...")
    if HAS_TRANSFORMERS:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print(f"✓ Loaded tokenizer: {args.model_name}")
    else:
        tokenizer = None
        print("⚠ Tokenizer not available - running in basic mode")
    
    # Tokenize data
    print("\n5. Tokenizing data...")
    dataset = tokenize_data(training_pairs, tokenizer, args.max_input_length, args.max_target_length)
    
    # Create train/val split
    print("\n6. Creating train/validation split...")
    dataset_dict = create_train_val_split(dataset, args.val_ratio)
    
    # Save processed data
    print("\n7. Saving processed data...")
    save_processed_data(dataset_dict, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✓ AskCQ preprocessing complete!")
    print(f"✓ Training examples: {len(dataset_dict['train'])}")
    print(f"✓ Validation examples: {len(dataset_dict['validation'])}")
    print(f"✓ Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()