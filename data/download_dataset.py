import json
import os
from datasets import load_dataset


def main():
    # Download the MS MARCO dataset (v1.1, train split)
    print("Downloading MS MARCO dataset...")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    # print first 5 samples
    for i, sample in enumerate(dataset):
        if i < 5:
            print(sample)

    # Convert the dataset to a dictionary
    dataset_dict = dataset.to_dict()

    # Save the dataset to ms_marco.json in the current folder (/data)
    output_filename = "ms_marco.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, ensure_ascii=False, indent=2)

    print(f"Dataset successfully saved to {output_filename}.")

if __name__ == "__main__":
    main()
