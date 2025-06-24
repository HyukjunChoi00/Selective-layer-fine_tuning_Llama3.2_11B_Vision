# Model
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# LoRA Config

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,  
    finetune_language_layers   = True,   
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

# fine-tuning cross-attention layer (q,v) 
for name, param in model.named_parameters():
    if 'cross_attn.q' in name or 'cross_attn.v' in name:
        param.requires_grad = True
for name, param in model.named_parameters():
    if 'cross_attn' in name and param.requires_grad:
        print(name)
# Load Dataset

from huggingface_hub import login

login(token='YOUR TOKEN')
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
converted_val = [convert_to_conversation(sample) for sample in val_dataset]

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    eval_dataset= converted_val,
    args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        #warmup_steps = 10,
        warmup_ratio = 0.05,
        #max_steps = 50,
        num_train_epochs = 3, # Set this instead of max_steps for full training runs
        learning_rate = 1e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        eval_strategy = 'steps',
        eval_steps = 500,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",  
        greater_is_better = False,           
        report_to = "wandb",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

import wandb

wandb.init(
    project="0512_llam_3ep_crossatn_data9",  # change this
    name="0512_llam_3ep_crossatn_data9",  # change this
    config=trainer,
)

# Check Trainable params ratio
def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"\nTrainable params: {trainable:,}")
    print(f"Total params: {total:,}")
    print(f"Trainable ratio: {100 * trainable / total:.4f}%")
print_trainable_parameters(model)

# Train
trainer_stats = trainer.train()
# Online saving
ACCESS_TOKEN = 'YOUR HuggingFace Token'
model.push_to_hub('HYUKJUNCHOI/0512_llam_3ep_crossatn_data9', use_temp_dir=True, use_auth_token=ACCESS_TOKEN)
tokenizer.push_to_hub('HYUKJUNCHOI/0512_llam_3ep_crossatn_data9', use_temp_dir=True, use_auth_token=ACCESS_TOKEN)


# Evaluation with OpenAI
api_key = 'YOUR OpenAI API KEY'
from openai import OpenAI
import time
def clean_prediction(prediction):
    # 채팅 템플릿 제거
    if "assistant" in prediction.lower():
        prediction = prediction.split("assistant")[-1].strip()
    if "user" in prediction.lower():
        prediction = prediction.split("user")[-1].strip()
    return prediction
def evaluate_with_gpt(predictions, references, api_key):
    client = OpenAI(api_key=api_key)
    correct_count = 0
    total_count = len(predictions)

    print(f"\nGPT 평가 시작 (총 {total_count}개):")

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # 채팅 템플릿 제거
        pred = clean_prediction(pred)

        # GPT Prompt
        prompt = f"""
       Please evaluate whether the following two sentences convey the same meaning.
    Reference sentence: {ref}
    Predicted sentence: {pred}

    If the two sentences have the same meaning, output '1'. If they do not, output '0' only.
        """

        try:
            # GPT API 호출
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates if two sentences have the same meaning. Only respond with '1' for same meaning or '0' for different meaning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # 응답 파싱
            score = int(response.choices[0].message.content.strip())
            correct_count += score

            # 진행 상황 출력
            if (i + 1) % 10 == 0:
                print(f"처리 중: {i + 1}/{total_count}")
                print(f"현재 정확도: {correct_count/(i+1):.2%}")

            # API 호출 간격 조절
            time.sleep(1)

        except Exception as e:
            print(f"오류 발생 (예시 {i+1}): {str(e)}")
            continue

    # 최종 결과 계산
    accuracy = correct_count / total_count

    print("\n평가 결과:")
    print(f"총 예시 수: {total_count}")
    print(f"정답 수: {correct_count}")
    print(f"정확도: {accuracy:.2%}")

    return {
        'total_count': total_count,
        'correct_count': correct_count,
        'accuracy': accuracy
    }

def evaluate_vlm(model, tokenizer, dataset, api_key=None):
    model.eval()
    all_predictions = []
    all_references = []
    num_samples = len(dataset)
    # 랜덤 샘플 선택
    samples = dataset.shuffle(seed=42).select(range(num_samples))

    print(f"\n모델 예측 시작 (총 {num_samples}개):")

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            # ... 기존 모델 예측 코드 ...
                        # 입력 데이터 준비
            image = sample["image"]
            instructions = sample['instructions']
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": instructions}
                ]}
            ]

            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            # 모델 예측
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                use_cache=True,
                temperature=0.5,
                min_p=0.1
            )

            # 예측 텍스트 디코딩 및 정제
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = clean_prediction(prediction)

            # 결과 저장
            all_predictions.append(prediction)
            all_references.append(sample['descriptions'])

            if (idx + 1) % 10 == 0:
                print(f"처리 중: {idx + 1}/{num_samples}")

    # GPT로 평가
    if api_key:
        results = evaluate_with_gpt(all_predictions, all_references, api_key)
        return results
    else:
        print("API key가 제공되지 않아 GPT 평가를 수행할 수 없습니다.")
        return None

# 사용 예시
results = evaluate_vlm(model, tokenizer, test_dataset, api_key=api_key)

