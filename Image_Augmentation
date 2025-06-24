from huggingface_hub import login

login(token='Your HuggingFace Token')
from datasets import load_dataset
dataset = load_dataset("HYUKJUNCHOI/0326")
dataset_train = dataset["train"]

# Image Augmentation
from torchvision import transforms
from datasets import load_dataset, Dataset, concatenate_datasets
import random

# 증강 정의
augment = transforms.Compose([
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.97, 1.03)),
])


n_augments = 9

# 증강된 샘플 저장
augmented_rows = []

for example in dataset_train:
    for _ in range(n_augments):
        new_image = augment(example["image"])
        augmented_rows.append({
            "instructions": example["instructions"],
            "image": new_image,
            "descriptions": example["descriptions"]
        })

# Check Augmentation
import matplotlib.pyplot as plt

# augmented_rows의 첫 번째 데이터 가져오기
first_example = augmented_rows[23]

# 첫 번째 이미지 시각화
plt.imshow(first_example["image"])
plt.title(f"Description: {first_example['descriptions']}")
plt.axis("off")  # 축 없애기
plt.show()

from datasets import Dataset
augmented_dataset = Dataset.from_list(augmented_rows)

from datasets import concatenate_datasets
full_dataset = concatenate_datasets([dataset_train, augmented_dataset])

# Upload Dataset
from huggingface_hub import login
import os
login(token='Your Huggingface Token')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'Your Huggingface Token'
full_dataset.push_to_hub("HYUKJUNCHOI/0507dataaugmentedn9")
