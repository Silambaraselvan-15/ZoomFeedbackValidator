import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch  #deep learning framework

# Load CSV files
train_df = pd.read_csv("train.csv", skiprows=1, names=["text", "reason", "label"])
eval_df = pd.read_csv("eval.csv", skiprows=1, names=["text", "reason", "label"])

# Clean data
train_df = train_df[train_df["label"].notna()]
train_df = train_df[train_df["label"] != "label"]
eval_df = eval_df[eval_df["label"].notna()]
eval_df = eval_df[eval_df["label"] != "label"]
train_df["label"] = train_df["label"].astype(int)
eval_df["label"] = eval_df["label"].astype(int)

# Convert to HF dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], example["reason"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize)
eval_dataset = eval_dataset.map(tokenize)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
)

# Compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean()
    return {"accuracy": accuracy.item()}

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print(train_df['label'].value_counts())

trainer.train()

# Save model and tokenizer
model.save_pretrained("feedback_model")
tokenizer.save_pretrained("feedback_model")

# Zip the folder
import shutil
shutil.make_archive("feedback_model", 'zip', "feedback_model")

# Download the zip file
from google.colab import files
files.download("feedback_model.zip")

print("Training complete. Model zipped and ready to download.")