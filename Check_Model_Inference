# Load Dataset

from huggingface_hub import login

login(token='YOUR HuggingFace Toekn')
from datasets import load_dataset

dataset = load_dataset("HYUKJUNCHOI/0504dataaugmentedn3")
dataset_train = dataset['train']
train_val_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=42)

train_dataset = train_val_test["train"]
val_dataset = val_test["train"]
test_dataset = val_test["test"]

# Loading Model
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
model, tokenizer = FastVisionModel.from_pretrained(
    "Your Model", # Change this
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# Checking Model Inference

def clean_prediction(prediction):
    # 채팅 템플릿 제거
    if "assistant" in prediction.lower():
        prediction = prediction.split("assistant")[-1].strip()
    if "user" in prediction.lower():
        prediction = prediction.split("user")[-1].strip()
    return prediction

# Change num_samples as you want
def evaluate_vlm(model, tokenizer, dataset, num_samples=10):
    model.eval()
    all_predictions = []
    all_references = []


    # 랜덤 샘플 선택
    samples = dataset.shuffle(seed=42).select(range(num_samples))

    print(f"\n평가 시작 (총 {num_samples}개):")

    with torch.no_grad():
        for idx, sample in enumerate(samples):
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
                max_new_tokens=300,
                use_cache=True,
                temperature=0.5,
                min_p=0.1
            )

            # 예측 텍스트 디코딩
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = clean_prediction(prediction)

            # 참조 텍스트
            reference = sample['descriptions']

            # 결과 저장
            all_predictions.append(prediction)
            all_references.append(reference)

            # 진행 상황 출력
            if (idx + 1) % 10 == 0:
                print(f"처리 중: {idx + 1}/{num_samples}")

    # 상세 예시 출력
    print("\n상세 예시 (처음 10개):")
    for i in range(min(10, len(all_predictions))):
        print(f"\n예시 {i+1}:")
        print(f"Instruction: {samples[i]['instructions']}")
        print(f"참조: {all_references[i]}")
        print(f"예측: {all_predictions[i]}")




# 평가 실행
results = evaluate_vlm(model, tokenizer, test_dataset)
