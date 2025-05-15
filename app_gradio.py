import gradio as gr
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("feedback_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load eval dataset and extract unique reasons
eval_df = pd.read_csv("eval.csv", skiprows=1, names=["text", "reason", "label"])
eval_df = eval_df[eval_df["reason"].notna()]  # remove NaNs
reasons = sorted(eval_df["reason"].unique().tolist())  # unique sorted reasons

# Prediction function
def predict(text, reason):
    inputs = tokenizer(text, reason, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "Matched" if pred == 1 else "Not Matched"

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Feedback Text"),
        gr.Dropdown(choices=reasons, label="Select Reason")
    ],
    outputs="text",
    title="Zoom Feedback Validator",
    description="Check whether feedback and reason match using a transformer model."
)

iface.launch(share=True)