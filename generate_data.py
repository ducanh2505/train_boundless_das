from pathlib import Path
from pyvene import set_seed
from transformers import LlamaTokenizer
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from tutorial_price_tagging_utils import (
    bound_alignment_sampler,
    lower_bound_alignment_example_sampler,
)

DATA_DIR = Path.cwd() / "data"
BATCH_SIZE = 16

tokenizer = LlamaTokenizer.from_pretrained("sharpbai/alpaca-7b-merged")

set_seed(42)

###################
# data loaders
###################


def build_dataset(raw_split):
    return Dataset.from_dict(
        {
            "input_ids": raw_split[0],
            "source_input_ids": raw_split[1],
            "labels": raw_split[2],
            "intervention_ids": raw_split[3],  # we will not use this field
        }
    )


raw_data = bound_alignment_sampler(
    tokenizer, 10000, [lower_bound_alignment_example_sampler]
)

raw_train = (
    raw_data[0][:8000],
    raw_data[1][:8000],
    raw_data[2][:8000],
    raw_data[3][:8000],
)
raw_eval = (
    raw_data[0][8000:9000],
    raw_data[1][8000:9000],
    raw_data[2][8000:9000],
    raw_data[3][8000:9000],
)
raw_test = (
    raw_data[0][9000:],
    raw_data[1][9000:],
    raw_data[2][9000:],
    raw_data[3][9000:],
)

dataset = DatasetDict(
    {
        "train": build_dataset(raw_train),
        "eval": build_dataset(raw_eval),
        "test": build_dataset(raw_test),
    }
)
dataset.save_to_disk(DATA_DIR)
print(f"Saved dataset to {DATA_DIR}")

reloaded_dataset = load_from_disk(DATA_DIR).with_format("torch")
print(f"Reloaded dataset from {DATA_DIR}")

train_dataset = reloaded_dataset["train"]
eval_dataset = reloaded_dataset["eval"]
test_dataset = reloaded_dataset["test"]

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(
    "Dataloaders ready: "
    f"train={len(train_dataloader)}, "
    f"eval={len(eval_dataloader)}, "
    f"test={len(test_dataloader)} batches"
)
