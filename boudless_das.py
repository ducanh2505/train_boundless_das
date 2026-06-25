import torch
import time
from tqdm import tqdm, trange
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import hydra
from omegaconf import DictConfig



from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from pyvene import set_seed, count_parameters


def calculate_loss(logits, labels, model, vocab_size):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    for k, v in model.interventions.items():
        boundary_loss = 1.0 * v.intervention_boundaries.sum()
    loss += boundary_loss
    return loss


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


def simple_boundless_das_position_config(model_type, intervention_type, layer):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,              # layer
                intervention_type,  # intervention type
            ),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return config

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    experiment_config = cfg.experiment
    set_seed(experiment_config.random_seed)
    device = experiment_config.device
    DATA_DIR = experiment_config.data_dir
    BATCH_SIZE = experiment_config.batch_size
    model_name = experiment_config.model
    config = LlamaConfig.from_pretrained(model_name)
    llama = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
        config=config,
    )


    reloaded_dataset = load_from_disk(DATA_DIR).with_format("torch")
    train_dataset = reloaded_dataset["train"]
    eval_dataset = reloaded_dataset["eval"]
    test_dataset = reloaded_dataset["test"]

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



    pv_config = simple_boundless_das_position_config(
        type(llama), experiment_config.pyvene_config.intevention_type, experiment_config.pyvene_config.layer
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
    temperature_start = 50.0
    temperature_end = 0.1
    target_total_step = len(train_dataloader) * epochs
    temperature_schedule = (
        torch.linspace(temperature_start, temperature_end, target_total_step)
        .to(torch.bfloat16)
        .to(device)
    )

    epochs = 3
    gradient_accumulation_steps = 4
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
            b_s = inputs["input_ids"].shape[0]
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
                    print("save and push to HF")
                    intervenable.save(
                        "./models", 
                        save_to_hf_hub=True, 
                        hf_repo_name="ducanh2505/pv_alpaca-7b-merged"
                    )
            total_step += 1


if __name__=="__main__":
    main()