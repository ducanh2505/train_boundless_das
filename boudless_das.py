import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from tqdm import tqdm, trange
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import hydra
from omegaconf import DictConfig
from utils import (
    calculate_loss, 
    compute_metrics,
    simple_boundless_das_position_config
)
from pyvene import (
    IntervenableModel,
)
from transformers import LlamaForCausalLM, LlamaConfig

from pyvene import set_seed, count_parameters

def create_llama_model(model_name):
    config = LlamaConfig.from_pretrained(model_name)
    llama = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
        config=config,
    )
    return llama

def reload_dataset(data_dir, batch_size, splits = ['train']):
    reloaded_dataset = load_from_disk(data_dir).with_format("torch")
    split_data_loaders = {}
    for split in splits:
        dataset = reloaded_dataset[split]
        dataloader = DataLoader(dataset, batch_size=batch_size)
        split_data_loaders[split] = dataloader
    return split_data_loaders


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(experiment_config: DictConfig):
    print(experiment_config)
    set_seed(experiment_config.random_seed)
    BATCH_SIZE = experiment_config.batch_size

    MODEL_NAME = experiment_config.model_name
    llama = create_llama_model(MODEL_NAME)

    DATA_DIR = experiment_config.data_dir
    data_loaders = reload_dataset(data_dir=DATA_DIR, batch_size=BATCH_SIZE, splits=['train'])
    train_dataloader = data_loaders['train']

    
    pv_config, config_id = simple_boundless_das_position_config(
        type(llama), "block_output", experiment_config.layer
    )
    intervenable = IntervenableModel(pv_config, llama)
    vocab_size = intervenable.model_config.vocab_size
    epochs = experiment_config.epochs
    t_total = int(len(train_dataloader) * 3)
    warm_up_steps = 0.1 * t_total
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v.rotate_layer.parameters()}]
        optimizer_params += [{"params": v.intervention_boundaries, "lr": 1e-2}]
    optimizer = torch.optim.AdamW(optimizer_params, lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
    )
    device = experiment_config.device
    temperature_start = 50.0
    temperature_end = 0.1
    target_total_step = len(train_dataloader) * epochs
    temperature_schedule = (
        torch.linspace(temperature_start, temperature_end, target_total_step)
        .to(torch.bfloat16)
        .to(device)
    )

    gradient_accumulation_steps = experiment_config.gradient_accumulation_steps
    total_step = 0

    intervenable.set_temperature(temperature_schedule[total_step])
    intervenable.disable_model_gradients()
    intervenable.set_device(device)
    
    print("llama trainable parameters: ", count_parameters(intervenable.model))
    print("intervention trainable parameters: ", intervenable.count_parameters())
    intervenable.model.train() 
    train_iterator = trange(0, int(epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"]}],
                {"sources->base": 80},  # swap 80th token
            )
            eval_metrics = compute_metrics(
                [counterfactual_outputs.logits], [inputs["labels"]]
            )

            # loss and backprop
            loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"], intervenable, vocab_size)
            loss_str = round(loss.item(), 2)
            epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                if not (gradient_accumulation_steps > 1 and total_step == 0):
                    optimizer.step()
                    scheduler.step()
                    intervenable.set_zero_grad()
                    intervenable.set_temperature(temperature_schedule[total_step])
            total_step += 1
        
        print("save and push to HF")
        intervenable.save(
            "./models", 
            save_to_hf_hub=True, 
            hf_repo_name=f"ducanh2505/pv_alpaca-7b-merged_{config_id}"
        )


if __name__=="__main__":
    main()