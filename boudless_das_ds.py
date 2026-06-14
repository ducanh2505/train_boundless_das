import torch
import time
import deepspeed
import os
from deepspeed.accelerator import get_accelerator
from tqdm import tqdm, trange
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
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
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from pyvene import set_seed, count_parameters

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

deepspeed.init_distributed()
_local_rank = int(os.environ.get("LOCAL_RANK"))
get_accelerator().set_device(_local_rank)

DATA_DIR = "data"
BATCH_SIZE = 16

config = LlamaConfig.from_pretrained("sharpbai/alpaca-7b-merged")
llama = LlamaForCausalLM.from_pretrained(
    "sharpbai/alpaca-7b-merged",
    torch_dtype=torch.bfloat16, 
)
tokenizer = LlamaTokenizer.from_pretrained("sharpbai/alpaca-7b-merged")
llama.eval() 

set_seed(42)


reloaded_dataset = load_from_disk(DATA_DIR).with_format("torch")
train_dataset = reloaded_dataset["train"]
eval_dataset = reloaded_dataset["eval"]
test_dataset = reloaded_dataset["test"]

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



config = simple_boundless_das_position_config(
    type(llama), "block_output", 15
)
intervenable = IntervenableModel(config, llama)
vocab_size = intervenable.model_config.vocab_size
optimizer_params = []

for _, intervention in intervenable.interventions.items():
    optimizer_params.append(
        {"params": intervention.rotate_layer.parameters()}
    )
    optimizer_params.append(
        {
            "params": intervention.intervention_boundaries,
            "lr": 1e-2,
        }
    )

engine, optimizer, trainloader, _ = deepspeed.initialize(
    model=intervenable,
    model_parameters=optimizer_params,
    training_data=train_dataset,
    config="ds_config.json",

)

local_rank = engine.local_rank
local_device = get_accelerator().device_name(local_rank)
target_dtype = None
if engine.bfloat16_enabled():
    target_dtype = torch.bfloat16
elif engine.fp16_enabled():
    target_dtype = torch.half


# t_total = int(len(train_dataloader) * 3)
# warm_up_steps = 0.1 * t_total
# optimizer_params = []
# for k, v in intervenable.interventions.items():
#     optimizer_params += [{"params": v.rotate_layer.parameters()}]
#     optimizer_params += [{"params": v.intervention_boundaries, "lr": 1e-2}]
# optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
# )



epochs = 3
gradient_accumulation_steps = 4
total_step = 0
target_total_step = len(train_dataloader) * epochs
temperature_start = 50.0
temperature_end = 0.1
temperature_schedule = (
    torch.linspace(temperature_start, temperature_end, target_total_step)
    .to(torch.bfloat16)
    .to(local_device)
)
engine.module.set_temperature(temperature_schedule[total_step])



def calculate_loss(logits, labels, engine, vocab_size):
    # Handle DeepSpeed wrapper
    intervenable = engine.module
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    for k, v in intervenable.interventions.items():
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



train_iterator = trange(0, int(epochs), desc="Epoch")
for epoch in train_iterator:
    epoch_iterator = tqdm(
        train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True, disable=local_rank!=0
    )
    for step, inputs in enumerate(epoch_iterator):
        print("sleep before move inputs to device")
        time.sleep(60)
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(local_device)
        b_s = inputs["input_ids"].shape[0]
        print("sleep after move inputs to device")
        time.sleep(60)
        _, counterfactual_outputs = engine(
            {"input_ids": inputs["input_ids"]},
            [{"input_ids": inputs["source_input_ids"]}],
            {"sources->base": 80},  # swap 80th token
        )
        eval_metrics = compute_metrics(
            [counterfactual_outputs.logits], [inputs["labels"]]
        )

        # loss and backprop
        loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"], engine, vocab_size)
        loss_str = round(loss.item(), 2)
        epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})

        engine.backward(loss)
        engine.step()
        engine.module.set_temperature(temperature_schedule[total_step])
        total_step += 1


# evaluation on the test set
eval_labels = []
eval_preds = []
with torch.no_grad():
    epoch_iterator = tqdm(test_dataloader, desc=f"Test")
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")
        b_s = inputs["input_ids"].shape[0]
        _, counterfactual_outputs = intervenable(
            {"input_ids": inputs["input_ids"]},
            [{"input_ids": inputs["source_input_ids"]}],
            {"sources->base": 80},  # swap 80th token
        )
        eval_labels += [inputs["labels"]]
        eval_preds += [counterfactual_outputs.logits]
eval_metrics = compute_metrics(eval_preds, eval_labels)
print(eval_metrics)

