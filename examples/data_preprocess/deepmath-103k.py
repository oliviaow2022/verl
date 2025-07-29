"""
Preprocess the DeepMath-103K dataset to parquet format
"""

import argparse
import os
import re
import shutil
import datasets
from sklearn.model_selection import train_test_split
# from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/scratch/rl_expts/data/DeepMath-103K")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation split")

    args = parser.parse_args()

    data_source = "zwhe99/DeepMath-103K"

    dataset = datasets.load_dataset(data_source)

    full_train_dataset  = dataset["train"]

    instruction_following = 'Let\'s think step by step and output your final answer inside \\boxed{{}}.'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction_following
            answer = example.pop("r1_solution_1")
            solution = example.pop("final_answer")
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    full_train_df = full_train_dataset.to_pandas()
    train_df, val_df = train_test_split(full_train_df, test_size=args.val_ratio, random_state=42)

    # Convert back to Huggingface Dataset
    train_dataset = datasets.Dataset.from_pandas(train_df)
    val_dataset = datasets.Dataset.from_pandas(val_df)
    
    # Apply the map function to both splits
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)
    
    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    print("Train Dataset saved")
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
    print("Val Dataset saved")