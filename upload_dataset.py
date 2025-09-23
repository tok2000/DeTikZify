import argparse
import os
from datasets import load_from_disk
from huggingface_hub import create_repo, upload_file, login

def main():
    parser = argparse.ArgumentParser(description="Convert Arrow dataset to Parquet and upload to HF Hub")
    parser.add_argument("--dataset_path", required=True, help="Path to local dataset folder (arrow format)")
    parser.add_argument("--hf_repo_id", required=True, help="Hugging Face dataset repo ID")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    repo_id = args.hf_repo_id

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # Convert to Parquet
    parquet_file = "dataset.parquet"
    print(f"Saving dataset as Parquet: {parquet_file}")
    dataset.to_parquet(parquet_file)

    # Create or connect to the Hugging Face dataset repo
    print(f"Creating or accessing Hugging Face dataset repo: {repo_id}")
    create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload the Parquet file
    print("Uploading the dataset file to Hugging Face Hub...")
    upload_file(
        path_or_fileobj=parquet_file,
        path_in_repo=parquet_file,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload dataset in Parquet format"
    )

    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
