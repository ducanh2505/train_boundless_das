import torch
from tqdm import tqdm
from datasets import load_from_disk
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
import pyvene as pv
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils import simple_boundless_das_position_config


# You can define your custom compute_metrics function.
def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -1]
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
        correct_labels = actual_test_labels == pred_test_labels
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
    accuracy = round(correct_count / total_count, 2)
    return {"accuracy": accuracy}

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(experiment_config: DictConfig):

    DATA_DIR = experiment_config.data_dir
    BATCH_SIZE = experiment_config.batch_size
    model_name = experiment_config.model_name
    device = experiment_config.device
    config = LlamaConfig.from_pretrained(model_name)
    llama = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
        config=config,
    )

    _, config_id = simple_boundless_das_position_config(
        type(llama), "block_output", experiment_config.layer
    )
    intervenable = pv.IntervenableModel.load(
        f"ducanh2505/pv_alpaca-7b-merged_{config_id}",
        model=llama,
        from_huggingface_hub=True

    )
    intervenable.set_device(device)

    reloaded_dataset = load_from_disk(DATA_DIR).with_format("torch")
    test_dataset = reloaded_dataset["test"]

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    # evaluation on the test set
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, desc=f"Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            b_s = inputs["input_ids"].shape[0]
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"]}],
                {"sources->base": 80},  # swap 80th token
            )
            eval_labels += [inputs["labels"]]
            eval_preds += [counterfactual_outputs.logits]
            if step == 10:
                break
    eval_metrics = compute_metrics(eval_preds, eval_labels)
    print(eval_metrics)

if __name__=="__main__":
    main()