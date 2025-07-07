# Imports
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import re
from colorama import Fore, Style
import nltk
import torch.backends.cuda as cuda

# Initialize colorama
from colorama import init
init()

cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Clear cache if needed
CLEAR_CACHE = False
if CLEAR_CACHE:
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    if os.path.exists(cache_dir):
        print("Clearing dataset cache...")
        shutil.rmtree(cache_dir, ignore_errors=True)

# Load & combine multiple datasets
SOURCES = [
    ("code_search_net", "javascript"),
    ("code_search_net", "python"),
    ("code_x_glue_ct_code_to_text", "javascript"),
    ("code_x_glue_ct_code_to_text", "python"),
]

# Extraction function remains the same
def extract_code_and_doc(example):
    # support both CodeSearchNet and CodeXGLUE fields
    code = example.get("func_code_string", example.get("code", "")).strip()
    doc  = example.get("func_documentation_string", example.get("docstring", example.get("doc", ""))).strip()
    if not code or not doc:
        return None
    if isinstance(code, list): code = " ".join(code)
    if isinstance(doc, list):  doc = " ".join(doc)
    return {"code": code, "doc": doc}

all_splits = []
for name, cfg in SOURCES:
    raw = load_dataset(name, cfg)
    for split in ("train", "validation"):
        ds = raw[split]
        ds = ds.map(extract_code_and_doc, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: x is not None)
        all_splits.append(ds)

# Concatenate and resplit
combined = concatenate_datasets(all_splits)
combined = combined.shuffle(seed=42)
splits = combined.train_test_split(test_size=0.1, seed=42)
train_raw, val_raw = splits["train"], splits["test"]
print(f"Combined train: {len(train_raw)}, validation: {len(val_raw)}")

# Tokenizer + Model
MODEL_NAME = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.use_cache = False

# Preprocess
def preprocess(examples):
    inputs = tokenizer(["summarize: " + c for c in examples["code"]],
                      max_length=512, truncation=True, padding="max_length")
    normalized_docs = []
    for doc in examples["doc"]:
        d = re.sub(r'\s+', ' ', doc).strip()
        normalized_docs.append(d)
    labels = tokenizer(normalized_docs,
                      max_length=128, truncation=True, padding="max_length")
    labels["input_ids"] = [[(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
                             for seq in labels["input_ids"]]
    return {"input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]}

# Tokenize combined sets
tokenized_train = train_raw.map(preprocess, batched=True, remove_columns=["code","doc"])
tokenized_val   = val_raw.map(preprocess,   batched=True, remove_columns=["code","doc"])

# Larger Subset for Better Learning
train_data = tokenized_train.select(range(min(15000, len(tokenized_train))))
val_data   = tokenized_val.select(range(min(2000,  len(tokenized_val))))

# Checkpoint
output_dir = Path("./code2doc_model")
checkpoints = list(output_dir.glob("checkpoint-*/"))
resume_ckpt = str(sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]) if checkpoints else None
print("Resuming from checkpoint:" if resume_ckpt else "Starting fresh training", resume_ckpt or "")

# Metrics + Logging Setup
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
log_dir = f"./output/training/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "metrics.tsv")
plot_file = os.path.join(log_dir, "loss_curve.png")
rogue_plot_file = os.path.join(log_dir, "rogueScore.png")
bleu_plot_file = os.path.join(log_dir, "bleuScore.png")
with open(log_file, "w") as f:
    f.write("Epoch\tTraining Loss\tValidation Loss\tBLEU\tROUGE-L\tSamples\n")

nltk.download("punkt_tab", quiet=True)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in labels back to pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = np.clip(labels, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Normalize whitespace
    decoded_preds = [re.sub(r'\s+', ' ', p).strip() for p in decoded_preds]
    decoded_labels = [re.sub(r'\s+', ' ', l).strip() for l in decoded_labels]

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(p)) for p in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(l)) for l in decoded_labels]

    # Compute ROUGE
    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
    rougeL = rouge_result.get("rougeLsum", 0.0) * 100

    # Compute BLEU
    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels],
    )
    bleu_score = bleu_result.get("bleu", 0.0) * 100

    return {
        "rougeL": rougeL,
        "bleu": bleu_score
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./code2doc_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    warmup_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=1,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    report_to=[],
    fp16=True,
    tf32=True,
    optim="adafactor",
    gradient_checkpointing=True,
    eval_accumulation_steps=2,
    group_by_length=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train
torch.cuda.empty_cache()
trainer.train(resume_from_checkpoint=resume_ckpt)
trainer.save_model("./code2doc_model_final")
tokenizer.save_pretrained("./code2doc_model_final")

# Extract logs and write to TSV
logs = trainer.state.log_history
train_loss = [x["loss"] for x in logs if "loss" in x]
eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
bleu_scores = [x["eval_bleu"] for x in logs if "eval_bleu" in x]
rouge_scores = [x["eval_rougeL"] for x in logs if "eval_rougeL" in x]
epochs = list(range(1, len(eval_loss) + 1))

for i, epoch in enumerate(epochs):
    metrics = [x for x in logs if x.get("epoch") == epoch and "eval_loss" in x]
    if metrics:
        m = metrics[0]
        
        print(f"\n{Fore.GREEN}=== Epoch {epoch} Metrics ===")
        print(f"{Fore.CYAN}Training Loss:{Style.RESET_ALL} {train_loss[i]:.4f}")
        print(f"{Fore.CYAN}Validation Loss:{Style.RESET_ALL} {m['eval_loss']:.4f}")
        print(f"{Fore.YELLOW}BLEU Score:{Style.RESET_ALL} {m.get('eval_bleu', 0):.2f}")
        print(f"{Fore.YELLOW}ROUGE-L Score:{Style.RESET_ALL} {m.get('eval_rougeL', 0):.2f}")
        
        if 'samples' in m:
            print(f"\n{Fore.BLUE}Sample Predictions:{Style.RESET_ALL}")
            for sample in m['samples']:
                print(f" - {sample}")
        
        # Write to file
        with open(log_file, "a") as f:
            f.write(f"{epoch}\t{train_loss[i]:.4f}\t{m['eval_loss']:.4f}\t{m.get('eval_bleu', 0):.2f}\t{m.get('eval_rougeL', 0):.2f}\t{'; '.join(m.get('samples', []))}\n")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss[:len(epochs)], label="Training Loss")
plt.plot(epochs, eval_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.savefig(plot_file)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs, rouge_scores, label="ROUGE-L Score", marker='s')
plt.xlabel("Epoch")
plt.ylabel("ROUGE-L Score")
plt.title("ROUGE-L Scores per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(rogue_plot_file)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs, bleu_scores, label="BLEU Score", marker='o')
plt.xlabel("Epoch")
plt.ylabel("BLEU Score")
plt.title("BLEU Scores per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(bleu_plot_file)
plt.close()

print(f"\n{Fore.GREEN}Training complete! Results saved to {log_dir}{Style.RESET_ALL}")