import datasets
from pathlib import Path
import json

def ensure_data_dir():
    """Ensures the data directory exists."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def download_humaneval():
    """Downloads and returns the HumanEval dataset."""
    print("Downloading HumanEval dataset...")
    try:
        dataset = datasets.load_dataset("openai_humaneval", split="test")
        print(f"✓ HumanEval dataset loaded successfully ({len(dataset)} samples)")
        return dataset
    except Exception as e:
        print(f"✗ Failed to download HumanEval dataset: {e}")
        return None

def download_mbpp():
    """Downloads and returns the MBPP dataset."""
    print("Downloading MBPP dataset...")
    try:
        # Try the most likely dataset names
        possible_names = [
            "google-research-datasets/mbpp",
            "mbpp",
            "google/mbpp", 
            "austin/mbpp"
        ]
        
        for name in possible_names:
            try:
                dataset = datasets.load_dataset(name, split="test")
                print(f"✓ MBPP dataset loaded successfully from {name} ({len(dataset)} samples)")
                return dataset
            except:
                continue
        
        # If none work, try with different split
        for name in possible_names:
            try:
                dataset = datasets.load_dataset(name, split="train")
                print(f"✓ MBPP dataset loaded successfully from {name} (train split) ({len(dataset)} samples)")
                return dataset
            except:
                continue
                
        raise Exception("None of the expected dataset names worked")
        
    except Exception as e:
        print(f"✗ Failed to download MBPP dataset: {e}")
        return None

def download_askcq():
    """Downloads and returns the AskCQ-like dataset (AmbigQA)."""
    print("Downloading AskCQ-like dataset (AmbigQA)...")
    # Using AmbigQA as a substitute for AskCQ - contains ambiguous questions requiring clarification
    try:
        dataset = datasets.load_dataset("sewon/ambig_qa", split="train")
        print(f"✓ AmbigQA dataset loaded successfully ({len(dataset)} samples)")
        return dataset
    except Exception as e:
        print(f"✗ Failed to download AmbigQA dataset: {e}")
        print("Note: Using AmbigQA as substitute for AskCQ dataset")
        return None

def create_sample_dataset(name, data_dir):
    """Create a minimal sample dataset when download fails."""
    sample_file = data_dir / f"{name}_sample.json"
    
    if name == "humaneval":
        sample_data = [{
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(numbers, threshold):\n    \"\"\" Check if in given list of numbers, are there any two numbers closer to each other than\n    given threshold.\n    \"\"\"",
            "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False",
            "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False"
        }]
    elif name == "mbpp":
        sample_data = [{
            "task_id": 1,
            "text": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
            "code": "R = 3\nC = 3\ndef min_cost(cost, m, n): \n\ttc = [[0 for x in range(C)] for x in range(R)] \n\ttc[0][0] = cost[0][0] \n\tfor i in range(1, m + 1): \n\t\ttc[i][0] = tc[i - 1][0] + cost[i][0] \n\tfor j in range(1, n + 1): \n\t\ttc[0][j] = tc[0][j - 1] + cost[0][j] \n\tfor i in range(1, m + 1): \n\t\tfor j in range(1, n + 1): \n\t\t\ttc[i][j] = min(tc[i - 1][j], tc[i][j - 1]) + cost[i][j] \n\treturn tc[m][n]"
        }]
    else:  # askcq
        sample_data = [{
            "id": "sample_1",
            "question": "What is the best programming language?",
            "annotations": [
                {"type": "ambiguous", "answer": "For what purpose? Web development, data science, or mobile apps?"}
            ]
        }]
    
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✓ Created sample dataset at {sample_file}")
    return sample_data

def save_dataset_info(dataset, name, data_dir):
    """Save basic dataset info to a text file."""
    if dataset is None:
        return
    
    info_file = data_dir / f"{name}_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"Number of samples: {len(dataset)}\n")
        if hasattr(dataset, 'features'):
            f.write(f"Features: {list(dataset.features.keys())}\n")
            f.write(f"Sample columns: {dataset.column_names}\n")
        else:
            f.write(f"Sample data structure: {type(dataset)}\n")
    print(f"✓ Dataset info saved to {info_file}")

def preprocess_humaneval(dataset, data_dir):
    """Preprocess HumanEval dataset into unified format."""
    if dataset is None:
        return None
        
    print("Preprocessing HumanEval dataset...")
    processed_data = []
    
    for item in dataset:
        if isinstance(item, dict):
            # Standard HumanEval format from Hugging Face
            unified_item = {
                "id": item.get("task_id", "unknown"),
                "prompt": item.get("prompt", ""),
                "canonical_solution": item.get("canonical_solution", ""),
                "test": item.get("test", ""),
                "entry_point": item.get("entry_point", ""),
                "dataset": "humaneval"
            }
        else:
            # Fallback for sample data format
            unified_item = {
                "id": item.get("task_id", "unknown"),
                "prompt": item.get("prompt", ""),
                "canonical_solution": item.get("canonical_solution", ""),
                "test": item.get("test", ""),
                "entry_point": "",
                "dataset": "humaneval"
            }
        processed_data.append(unified_item)
    
    # Save as JSONL
    output_file = data_dir / "humaneval.jsonl"
    with open(output_file, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ HumanEval dataset preprocessed and saved to {output_file}")
    return processed_data

def preprocess_mbpp(dataset, data_dir):
    """Preprocess MBPP dataset into unified format."""
    if dataset is None:
        return None
        
    print("Preprocessing MBPP dataset...")
    processed_data = []
    
    for item in dataset:
        if isinstance(item, dict):
            # Standard MBPP format
            unified_item = {
                "id": str(item.get("task_id", item.get("id", "unknown"))),
                "prompt": item.get("text", item.get("prompt", "")),
                "canonical_solution": item.get("code", item.get("canonical_solution", "")),
                "test": item.get("test_list", item.get("test", "")),
                "entry_point": "",  # MBPP doesn't typically have entry points
                "dataset": "mbpp"
            }
        else:
            # Fallback for sample data format
            unified_item = {
                "id": str(item.get("task_id", "unknown")),
                "prompt": item.get("text", ""),
                "canonical_solution": item.get("code", ""),
                "test": "",
                "entry_point": "",
                "dataset": "mbpp"
            }
        processed_data.append(unified_item)
    
    # Save as JSONL
    output_file = data_dir / "mbpp.jsonl"
    with open(output_file, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ MBPP dataset preprocessed and saved to {output_file}")
    return processed_data

def preprocess_askcq(dataset, data_dir):
    """Preprocess AskCQ/AmbigQA dataset into unified format."""
    if dataset is None:
        return None
        
    print("Preprocessing AskCQ dataset...")
    processed_data = []
    
    for item in dataset:
        if isinstance(item, dict):
            # AmbigQA or sample format
            unified_item = {
                "id": str(item.get("id", item.get("sample_id", "unknown"))),
                "question": item.get("question", ""),
                "annotations": item.get("annotations", []),
                "answers": item.get("answers", []),
                "dataset": "askcq"
            }
        else:
            # Fallback format
            unified_item = {
                "id": "unknown",
                "question": str(item) if isinstance(item, str) else "",
                "annotations": [],
                "answers": [],
                "dataset": "askcq"
            }
        processed_data.append(unified_item)
    
    # Save as JSONL
    output_file = data_dir / "askcq.jsonl"
    with open(output_file, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ AskCQ dataset preprocessed and saved to {output_file}")
    return processed_data

def main():
    """Main function to download all datasets."""
    print("=" * 60)
    print("UMinFramework - Dataset Download Script")
    print("=" * 60)
    
    # Ensure data directory exists
    data_dir = ensure_data_dir()
    print(f"Data directory: {data_dir.absolute()}")
    
    # Download all datasets
    datasets_info = []
    
    # HumanEval
    print("\n" + "-" * 40)
    humaneval_data = download_humaneval()
    if humaneval_data:
        save_dataset_info(humaneval_data, "humaneval", data_dir)
        processed_humaneval = preprocess_humaneval(humaneval_data, data_dir)
        datasets_info.append(("HumanEval", len(processed_humaneval) if processed_humaneval else 0, "✓"))
    else:
        print("Creating sample HumanEval dataset as fallback...")
        humaneval_data = create_sample_dataset("humaneval", data_dir)
        save_dataset_info(humaneval_data, "humaneval", data_dir)
        processed_humaneval = preprocess_humaneval(humaneval_data, data_dir)
        datasets_info.append(("HumanEval (sample)", len(processed_humaneval) if processed_humaneval else 0, "⚠"))

    # MBPP
    print("\n" + "-" * 40)
    mbpp_data = download_mbpp()
    if mbpp_data:
        save_dataset_info(mbpp_data, "mbpp", data_dir)
        processed_mbpp = preprocess_mbpp(mbpp_data, data_dir)
        datasets_info.append(("MBPP", len(processed_mbpp) if processed_mbpp else 0, "✓"))
    else:
        print("Creating sample MBPP dataset as fallback...")
        mbpp_data = create_sample_dataset("mbpp", data_dir)
        save_dataset_info(mbpp_data, "mbpp", data_dir)
        processed_mbpp = preprocess_mbpp(mbpp_data, data_dir)
        datasets_info.append(("MBPP (sample)", len(processed_mbpp) if processed_mbpp else 0, "⚠"))

    # AskCQ (AmbigQA)
    print("\n" + "-" * 40)
    askcq_data = download_askcq()
    if askcq_data:
        save_dataset_info(askcq_data, "askcq", data_dir)
        processed_askcq = preprocess_askcq(askcq_data, data_dir)
        datasets_info.append(("AskCQ (AmbigQA)", len(processed_askcq) if processed_askcq else 0, "✓"))
    else:
        print("Creating sample AskCQ dataset as fallback...")
        askcq_data = create_sample_dataset("askcq", data_dir)
        save_dataset_info(askcq_data, "askcq", data_dir)
        processed_askcq = preprocess_askcq(askcq_data, data_dir)
        datasets_info.append(("AskCQ (sample)", len(processed_askcq) if processed_askcq else 0, "⚠"))
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, count, status in datasets_info:
        print(f"{status} {name:<20} {count:>8} samples")
    
    successful_downloads = sum(1 for _, _, status in datasets_info if status == "✓")
    sample_fallbacks = sum(1 for _, _, status in datasets_info if status == "⚠")
    print(f"\nSuccessfully downloaded {successful_downloads}/3 datasets")
    if sample_fallbacks > 0:
        print(f"Created {sample_fallbacks} sample datasets as fallbacks")
        print("\nNote: Sample datasets are minimal and should be replaced with full datasets for production use.")
    
    return {
        'humaneval': {
            'raw': humaneval_data,
            'processed': processed_humaneval if 'processed_humaneval' in locals() else None
        },
        'mbpp': {
            'raw': mbpp_data,
            'processed': processed_mbpp if 'processed_mbpp' in locals() else None
        },
        'askcq': {
            'raw': askcq_data,
            'processed': processed_askcq if 'processed_askcq' in locals() else None
        }
    }

if __name__ == "__main__":
    main()
