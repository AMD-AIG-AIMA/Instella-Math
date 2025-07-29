"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
import datasets
from typing import Dict, List, Optional, Any
import re
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from deepscaler.data.utils import load_dataset
from deepscaler.data.dataset_types import TrainDataset, TestDataset


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))

def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('question')
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{question} {instruction}"
        answer = example.pop('answer')
        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('./deepscaler/data_processed'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)

    data_source = 'opencompass/AIME2025'

    dataset_1 = datasets.load_dataset(data_source, 'AIME2025-I')
    test_dataset_1 = dataset_1['test']
    dataset_2 = datasets.load_dataset(data_source, 'AIME2025-II')
    test_dataset_2 = dataset_2['test']

    test_dataset = datasets.concatenate_datasets([test_dataset_1,test_dataset_2])

    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    test_dataset.to_parquet(os.path.join(local_dir, 'aime25.parquet'))
    print(f"aime25 test data size:", len(test_dataset))