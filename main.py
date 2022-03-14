import datasets
from typing import Dict
import pandas as pd


class GLUELoader:
    """Simple class for loading GLUE datasets as Pandas DataFrames."""

    def __init__(self):
        self.datasets = [
            "ax",
            "cola",
            "mnli",
            "mrpc",
            "qnli",
            "qqp",
            "rte",
            "sst2",
            "stsb",
            "wnli",
        ]

    def load_dataset(self, dataset_name: str) -> Dict[str, pd.DataFrame]:
        """Load glue dataset."""

        assert (
            dataset_name in self.datasets
        ), "{} is not a GLUE dataset. Find all GLUE datasets here: https://huggingface.co/datasets/glue"

        hf_dataset = datasets.load_dataset("glue", dataset_name)
        splits = list(hf_dataset.column_names.keys())

        dataset = {}
        for split in splits:
            dataset[split] = hf_dataset[split].to_pandas()

        return dataset
