from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
from huggingface_hub import login

login(token='YOUR HuggingFace Token')
from datasets import load_dataset

dataset = load_dataset("HYUKJUNCHOI/0507dataaugmentedn9")
dataset_train = dataset['train']
train_val_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=42)

train_dataset = train_val_test["train"]
val_dataset = val_test["train"]
test_dataset = val_test["test"]
def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : sample['instructions']},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["descriptions"]} ]
        },
    ]
    return { "messages" : conversation }
pass
converted_dataset = [convert_to_conversation(sample) for sample in train_dataset]

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,  # 비전 레이어 프리징
    finetune_language_layers   = True,   # 언어 레이어 파인튜닝
    finetune_attention_modules = False,
    finetune_mlp_modules       = True,
    r = 32,
    lora_alpha = 64,
    lora_dropout = 0.1,
    bias = "none",
    random_state = 3407,
    use_rslora = True,
    loftq_config = None,
)
# Fine-tune Vision Model LayerNorm
for name, param in model.named_parameters():
    if 'vision_model' in name and 'layernorm' in name:
        param.requires_grad = True
for name, param in model.named_parameters():
    if 'vision_model' in name and param.requires_grad:
        print(name)

#Training Config

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.05,
        max_steps = 50,
        #num_train_epochs = 3, # Set this instead of max_steps for full training runs
        learning_rate = 1e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

pre_weights = {}
for name, param in model.named_parameters():
    if 'vision_model' in name and 'layernorm' in name.lower():  # LayerNorm 감지
        pre_weights[name] = param.detach().clone()

#Train
trainer_stats = trainer.train()

# Check LayerNorm weight Changes
import torch

layernorm_deltas = []

for name, param in model.named_parameters():
    if name in pre_weights:
        # squared L2 norm of the parameter change
        delta = torch.norm(pre_weights[name] - param).item() ** 2
        layernorm_deltas.append((name, delta))

# 변화량 기준 정렬
layernorm_deltas.sort(key=lambda x: x[1], reverse=True)

# 상위 6개 예시 출력
for name, delta in layernorm_deltas[:6]:
    print(f"{name}: 변화량 {delta:.6f}")
