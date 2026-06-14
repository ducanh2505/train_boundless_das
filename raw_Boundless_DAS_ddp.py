#!/usr/bin/env python
# coding: utf-8

import os
import math
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    BitsAndBytesConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.nn import CrossEntropyLoss
from tutorial_price_tagging_utils import (
    factual_sampler,
    bound_alignment_sampler,
    lower_bound_alignment_example_sampler,
)

from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import set_seed, count_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="sharpbai/alpaca-7b-merged")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--disable-fp16-autocast", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--prealign-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def create_llama(
    name="sharpbai/alpaca-7b-merged",
    cache_dir=None,
    device=None,
    load_in_4bit=False,
    load_in_8bit=False,
):
    if load_in_4bit and load_in_8bit:
        raise ValueError("Use only one of --load-in-4bit or --load-in-8bit.")

    config = LlamaConfig.from_pretrained(name, cache_dir=cache_dir)
    config.use_cache = False

    tokenizer = LlamaTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "config": config,
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit or load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": device.index}
    else:
        model_kwargs["torch_dtype"] = torch.float16

    llama = LlamaForCausalLM.from_pretrained(name, **model_kwargs)
    if not (load_in_4bit or load_in_8bit):
        llama.to(device)

    llama.config.use_cache = False
    llama.eval()
    print("loaded model")
    return config, tokenizer, llama


args = parse_args()

dist.init_process_group("nccl")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
rank = dist.get_rank()
world_size = dist.get_world_size()
use_fp16_autocast = not args.disable_fp16_autocast

config, tokenizer, llama = create_llama(
    name=args.model_name,
    cache_dir=args.cache_dir,
    device=device,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
)

print(f"Running on rank {rank} / {world_size} on local GPU {local_rank}")

raw_prealign = factual_sampler(tokenizer, 5000, game="pricing_tag")
prealign_dataset = Dataset.from_dict(
    {"input_ids": raw_prealign[0], "labels": raw_prealign[1]}
)
prealign_dataset.set_format("torch", columns=["input_ids", "labels"])
prealign_dataloader = DataLoader(
    prealign_dataset,
    batch_size=args.prealign_batch_size,
)

set_seed(42)


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
train_dataset = Dataset.from_dict(
    {
        "input_ids": raw_train[0],
        "source_input_ids": raw_train[1],
        "labels": raw_train[2],
        "intervention_ids": raw_train[3],  # we will not use this field
    }
).with_format("torch")
train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    sampler=train_sampler,
    pin_memory=True,
)
eval_dataset = Dataset.from_dict(
    {
        "input_ids": raw_eval[0],
        "source_input_ids": raw_eval[1],
        "labels": raw_eval[2],
        "intervention_ids": raw_eval[3],  # we will not use this field
    }
).with_format("torch")
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=args.eval_batch_size,
    pin_memory=True,
)
test_dataset = Dataset.from_dict(
    {
        "input_ids": raw_test[0],
        "source_input_ids": raw_test[1],
        "labels": raw_test[2],
        "intervention_ids": raw_test[3],  # we will not use this field
    }
).with_format("torch")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    pin_memory=True,
)



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


config = simple_boundless_das_position_config(
    type(llama), "block_output", 15
)
intervenable = IntervenableModel(config, llama)
intervenable.set_device(device)
intervenable.disable_model_gradients()
intervenable = DDP(
    intervenable,
    device_ids=[local_rank],
    output_device=local_rank,
)


epochs = args.epochs
gradient_accumulation_steps = args.gradient_accumulation_steps
t_total = math.ceil(len(train_dataloader) / gradient_accumulation_steps) * epochs
warm_up_steps = int(0.1 * t_total)
optimizer_params = []
for k, v in intervenable.module.interventions.items():
    optimizer_params += [{"params": v.rotate_layer.parameters()}]
    optimizer_params += [{"params": v.intervention_boundaries, "lr": 1e-2}]
optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
)


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


total_step = 0
target_total_step = len(train_dataloader) * epochs
temperature_start = 50.0
temperature_end = 0.1
temperature_schedule = (
    torch.linspace(temperature_start, temperature_end, target_total_step)
    .to(torch.float16)
    .to(device)
)
intervenable.module.set_temperature(temperature_schedule[total_step])


def calculate_loss(logits, labels):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, intervenable.module.model_config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    boundary_loss = 0.0
    for _, v in intervenable.module.interventions.items():
        boundary_loss = boundary_loss + v.intervention_boundaries.sum()
    loss += boundary_loss

    return loss


intervenable.module.model.train()  # train enables drop-off but no grads
if rank == 0:
    print("llama trainable parameters: ", count_parameters(intervenable.module.model))
    print("intervention trainable parameters: ", intervenable.module.count_parameters())
train_iterator = trange(0, int(epochs), desc="Epoch", disable=rank != 0)
for epoch in train_iterator:
    train_sampler.set_epoch(epoch)
    epoch_iterator = tqdm(
        train_dataloader,
        desc=f"Epoch: {epoch}",
        position=0,
        leave=True,
        disable=rank != 0,
    )
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device, non_blocking=True)

        is_last_batch = step == len(train_dataloader) - 1
        should_step = (total_step + 1) % gradient_accumulation_steps == 0 or is_last_batch
        sync_context = nullcontext() if should_step else intervenable.no_sync()
        with sync_context:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_fp16_autocast,
            ):
                _, counterfactual_outputs = intervenable(
                    {"input_ids": inputs["input_ids"]},
                    [{"input_ids": inputs["source_input_ids"]}],
                    {"sources->base": 80},  # swap 80th token
                )
                eval_metrics = compute_metrics(
                    [counterfactual_outputs.logits], [inputs["labels"]]
                )

                # loss and backprop
                loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"])
            loss_str = round(loss.item(), 2)
            epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

        if should_step:
            optimizer.step()
            scheduler.step()
            intervenable.module.set_zero_grad()
            intervenable.module.set_temperature(temperature_schedule[total_step])
        total_step += 1


eval_labels = []
eval_preds = []
dist.barrier()
with torch.no_grad():
    epoch_iterator = tqdm(test_dataloader, desc="Test", disable=rank != 0)
    for step, inputs in enumerate(epoch_iterator):
        if rank != 0:
            break
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device, non_blocking=True)
        b_s = inputs["input_ids"].shape[0]
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_fp16_autocast,
        ):
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"]}],
                {"sources->base": 80},  # swap 80th token
            )
        eval_labels += [inputs["labels"]]
        eval_preds += [counterfactual_outputs.logits]
if rank == 0:
    eval_metrics = compute_metrics(eval_preds, eval_labels)
    print(eval_metrics)

dist.barrier()
dist.destroy_process_group()
