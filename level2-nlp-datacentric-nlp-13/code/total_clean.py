import os
import random

import evaluate
import numpy as np
import pandas as pd
import torch
from cleanlab.filter import find_label_issues
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "split_train_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "split_train_data")

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=7)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터 로드
data = pd.read_csv(os.path.join(INPUT_DIR, "merge_text_label_clean.csv"))
dataset_train, dataset_valid = train_test_split(data, test_size=0.2, random_state=SEED)
texts = data["text"].tolist()
labels = data["target"].tolist()


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data["text"]
        targets = data["target"]
        self.inputs = []
        self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(
                text, padding="max_length", truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"].squeeze(0),
            "attention_mask": self.inputs[idx]["attention_mask"].squeeze(0),
            "labels": self.labels[idx].squeeze(0),
        }

    def __len__(self):
        return len(self.labels)


# 데이터셋 및 데이터로더 생성
data_train = BERTDataset(dataset_train, tokenizer)
data_valid = BERTDataset(dataset_valid, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="macro")


os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    logging_strategy="steps",
    eval_strategy="steps",
    save_strategy="no",
    logging_steps=100,
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    learning_rate=2e-05,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print("Fine-tuning 완료")


def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    model.eval()
    model.config.output_hidden_states = True
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="텍스트 임베딩 중"):
        batch = texts[i : i + batch_size]
        encoded_input = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=256
        )
        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
    normalized_embeddings = normalize(np.vstack(embeddings), norm="l2")
    return normalized_embeddings


embeddings = get_embeddings(texts, model, tokenizer, device)

n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

pred_probs = np.zeros((len(labels), 7))

for fold, (train_index, val_index) in enumerate(skf.split(embeddings, labels), 1):
    print(f"\n폴드 {fold}/{n_splits} 처리 중...")
    X_train, X_val = embeddings[train_index], embeddings[val_index]
    y_train, y_val = np.array(labels)[train_index], np.array(labels)[val_index]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    pred_probs[val_index] = clf.predict_proba(X_val)

print("\n라벨 오류 탐지 중...")
label_issues = find_label_issues(
    labels=labels,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence",
    filter_by="both",
    frac_noise=0.2,
    min_examples_per_class=10,
)

cleaned_labels = np.array(labels).copy()
cleaned_labels[label_issues] = np.argmax(pred_probs[label_issues], axis=1)

clean_df = data.copy()
clean_df["cleaned_target"] = cleaned_labels

different_labels = clean_df[clean_df["target"] != clean_df["cleaned_target"]]

print(f"\n변경된 라벨 수: {len(different_labels)}")

clean_df = clean_df.drop("target", axis=1)
clean_df = clean_df.rename(columns={"cleaned_target": "target"})

clean_df.to_csv(os.path.join(OUTPUT_DIR, "total_cleaned.csv"), index=False)

print("\n정제된 데이터가 total_cleaned.csv 파일로 저장되었습니다.")