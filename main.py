# ✅ Imports
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# ✅ Clear cache if needed
CLEAR_CACHE = False
if CLEAR_CACHE:
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    if os.path.exists(cache_dir):
        print("Clearing dataset cache...")
        shutil.rmtree(cache_dir, ignore_errors=True)

# ✅ Load dataset
try:
    dataset = load_dataset("code_search_net", "javascript")
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = load_dataset("code_search_net", "javascript", download_mode="force_redownload")

# ✅ Extract code + doc
def extract_code_and_doc(example):
    code = example.get("func_code_string", "").strip()
    doc  = example.get("func_documentation_string", "").strip()
    if not code or not doc:
        return None
    if isinstance(code, list): code = " ".join(code)
    if isinstance(doc, list):  doc  = " ".join(doc)
    return {"code": code, "doc": doc}

dataset = dataset.map(extract_code_and_doc)
dataset = dataset.filter(lambda x: x is not None)
dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in ("code", "doc")])

# ✅ Tokenizer + Model
tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
model     = T5ForConditionalGeneration.from_pretrained("t5-small")

# ✅ Preprocess
def preprocess(examples):
    inputs = tokenizer(["summarize: " + c for c in examples["code"]],
                       max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(examples["doc"],
                       max_length=64, truncation=True, padding="max_length")
    labels["input_ids"] = [[(tok if tok != tokenizer.pad_token_id else -100) for tok in seq] for seq in labels["input_ids"]]
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"]
    }

tokenized = dataset.map(preprocess, batched=True, remove_columns=["code", "doc"])

# ✅ Larger Subset for Better Learning
train_data = tokenized["train"].select(range(min(15000, len(tokenized["train"]))))
val_data   = tokenized["validation"].select(range(min(2000, len(tokenized["validation"]))))

# ✅ Checkpoint
output_dir = Path("./code2doc_model")
checkpoints = list(output_dir.glob("checkpoint-*/"))
resume_ckpt = str(sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]) if checkpoints else None
print("Resuming from checkpoint:" if resume_ckpt else "Starting fresh training", resume_ckpt or "")

# ✅ Metrics + Logging Setup
bleu  = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

log_dir = f"./output/training/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "metrics.tsv")
plot_file = os.path.join(log_dir, "loss_curve.png")

with open(log_file, "w") as f:
    f.write("Epoch\tTraining Loss\tValidation Loss\tBLEU\tROUGE-L\n")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.rint(predictions).astype(int)
    predictions = np.clip(predictions, 0, None)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels.astype(int), skip_special_tokens=True)
    bleu_res  = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    rouge_res = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": bleu_res["score"], "rougeL": rouge_res["rougeL"]}

# ✅ Training Args (Optimized for RTX 3050)
training_args = Seq2SeqTrainingArguments(
    output_dir="./code2doc_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    warmup_steps=200,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    report_to="tensorboard",
    fp16=True,
    gradient_accumulation_steps=1,
    group_by_length=True,
    gradient_checkpointing=True
)

# ✅ Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ✅ Train
train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
trainer.save_model("./code2doc_model_final")
tokenizer.save_pretrained("./code2doc_model_final")

# ✅ Extract logs and write to TSV
logs = trainer.state.log_history
train_loss = [x["loss"] for x in logs if "loss" in x]
eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
epochs = list(range(1, len(eval_loss) + 1))

for i, epoch in enumerate(epochs):
    metrics = [x for x in logs if x.get("epoch") == epoch and "eval_loss" in x]
    if metrics:
        m = metrics[0]
        with open(log_file, "a") as f:
            f.write(f"{epoch}\t{train_loss[i]:.4f}\t{m['eval_loss']:.4f}\t{m.get('bleu', 0):.2f}\t{m.get('rougeL', 0):.2f}\n")

# ✅ Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss[:len(epochs)], label="Training Loss")
plt.plot(epochs, eval_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig(plot_file)
plt.close()
