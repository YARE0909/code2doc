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
import torch
import re
from colorama import Fore, Style
import nltk
import torch.backends.cuda as cuda

# Initialize colorama
from colorama import init
init()

cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix mult
torch.backends.cudnn.benchmark = True  # Auto-tune CUDA kernels

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
    doc = example.get("func_documentation_string", "").strip()
    if not code or not doc:
        return None
    if isinstance(code, list): code = " ".join(code)
    if isinstance(doc, list): doc = " ".join(doc)
    return {"code": code, "doc": doc}

dataset = dataset.map(extract_code_and_doc)
dataset = dataset.filter(lambda x: x is not None)
dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in ("code", "doc")])

# ✅ Tokenizer + Model
MODEL_NAME = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.use_cache = False

# ✅ Preprocess
def preprocess(examples):
    # Prepend "summarize: " to code and normalize docs
    inputs = tokenizer(["summarize: " + c for c in examples["code"]],
                      max_length=512, truncation=True, padding="max_length")
    
    # Normalize documentation strings
    normalized_docs = []
    for doc in examples["doc"]:
        # Remove excessive whitespace and special characters
        doc = re.sub(r'\s+', ' ', doc).strip()
        normalized_docs.append(doc)
    
    labels = tokenizer(normalized_docs,
                      max_length=128, truncation=True, padding="max_length")
    labels["input_ids"] = [[(tok if tok != tokenizer.pad_token_id else -100) for tok in seq] for seq in labels["input_ids"]]
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"]
    }

tokenized = dataset.map(preprocess, batched=True, remove_columns=["code", "doc"])

# ✅ Larger Subset for Better Learning
train_data = tokenized["train"].select(range(min(15000, len(tokenized["train"]))))
val_data = tokenized["validation"].select(range(min(2000, len(tokenized["validation"]))))

# ✅ Checkpoint
output_dir = Path("./code2doc_model")
checkpoints = list(output_dir.glob("checkpoint-*/"))
resume_ckpt = str(sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]) if checkpoints else None
print("Resuming from checkpoint:" if resume_ckpt else "Starting fresh training", resume_ckpt or "")

# ✅ Metrics + Logging Setup
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

log_dir = f"./output/training/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "metrics.tsv")
plot_file = os.path.join(log_dir, "loss_curve.png")

with open(log_file, "w") as f:
    f.write("Epoch\tTraining Loss\tValidation Loss\tBLEU\tROUGE-L\tSamples\n")

def normalize_text(text):
    """Normalize text for metric comparison"""
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    text = re.sub(r'\s+', ' ', text).strip()     # Normalize whitespace
    return text

nltk.download("punkt_tab", quiet=True)

def compute_metrics(eval_pred):
    try:
        predictions, labels = eval_pred
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Convert to numpy arrays and ensure valid token IDs
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Clip to valid token IDs range
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
        
        # Decode with error handling
        decoded_preds = []
        for pred in predictions:
            try:
                decoded = tokenizer.decode(pred, skip_special_tokens=True)
                decoded_preds.append(decoded if decoded.strip() else " ")
            except:
                decoded_preds.append(" ")
        
        decoded_labels = []
        for label in labels:
            try:
                decoded = tokenizer.decode(label, skip_special_tokens=True)
                decoded_labels.append(decoded if decoded.strip() else " ")
            except:
                decoded_labels.append(" ")
        
        # ROUGE-LSum expects newline separated sentences
        try:
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        except:
            pass
        
        # Compute metrics
        metrics = {
            "rougeL": 0.0,
            "bleu": 0.0
        }
        
        try:
            rouge_result = rouge.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
            metrics["rougeL"] = rouge_result["rougeL"] * 100
        except Exception as e:
            print(f"ROUGE Error: {str(e)}")
        
        try:
            bleu_result = bleu.compute(
                predictions=decoded_preds,
                references=[[label] for label in decoded_labels],
                smooth_method="floor",
                smooth_value=0.1
            )
            metrics["bleu"] = bleu_result["score"]
        except Exception as e:
            print(f"BLEU Error: {str(e)}")
        
        # Print samples for debugging
        print("\nSample predictions vs references:")
        for i in range(min(3, len(decoded_preds))):
            print(f"Pred: {decoded_preds[i][:100]}...")
            print(f"Ref: {decoded_labels[i][:100]}...\n")
        
        return metrics
    
    except Exception as e:
        print(f"Overall metric computation failed: {str(e)}")
        return {
            "rougeL": 0.0,
            "bleu": 0.0
        }
    
# ✅ Training Args
training_args = Seq2SeqTrainingArguments(
    output_dir="./code2doc_model",
    eval_strategy="steps",          # More frequent than epoch
    eval_steps=200,                # Evaluate every 1000 steps
    save_strategy="steps",
    save_steps=1000,
    learning_rate=5e-4,             # Slightly higher for Adafactor
    warmup_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=1,         # Faster evaluation (was 4)
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    report_to="tensorboard",
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

# ✅ Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def run_test():
    try:
        input_text = "summarize: function test() { return 1; }"
        label_text = "Test function"
        
        print(f"\n{'='*50}\nTest Debugging\n{'='*50}")
        print(f"Input text: {input_text}")
        print(f"Label text: {label_text}")
        
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        labels = tokenizer(label_text, return_tensors="pt").input_ids.to(device)
        
        print("\nTokenized input IDs:", inputs["input_ids"])
        print("Tokenized labels:", labels)
        
        with torch.no_grad():
            test_preds = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=4,
                early_stopping=True
            )
        
        print("\nRaw predictions tensor:", test_preds)
        
        decoded_input = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        decoded_pred = tokenizer.decode(test_preds[0], skip_special_tokens=True)
        decoded_label = tokenizer.decode(labels[0], skip_special_tokens=True)
        
        print("\nDecoded input:", decoded_input)
        print("Decoded prediction:", decoded_pred)
        print("Decoded label:", decoded_label)
        
        test_preds = test_preds.cpu().numpy()
        labels = labels.cpu().numpy()
        
        metrics = compute_metrics((test_preds, labels))
        print("\nTest metrics:", metrics)
        return metrics
    
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return {"bleu": 0.0, "rougeL": 0.0}

# Run test before training
test_metrics = run_test()

# ✅ Train
torch.cuda.empty_cache()
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
        
        # Print colored output
        print(f"\n{Fore.GREEN}=== Epoch {epoch} Metrics ===")
        print(f"{Fore.CYAN}Training Loss:{Style.RESET_ALL} {train_loss[i]:.4f}")
        print(f"{Fore.CYAN}Validation Loss:{Style.RESET_ALL} {m['eval_loss']:.4f}")
        print(f"{Fore.YELLOW}BLEU Score:{Style.RESET_ALL} {m.get('bleu', 0):.2f}")
        print(f"{Fore.YELLOW}ROUGE-L Score:{Style.RESET_ALL} {m.get('rougeL', 0):.2f}")
        
        # Print sample predictions
        if 'samples' in m:
            print(f"\n{Fore.BLUE}Sample Predictions:{Style.RESET_ALL}")
            for sample in m['samples']:
                print(f" - {sample}")
        
        # Write to file
        with open(log_file, "a") as f:
            f.write(f"{epoch}\t{train_loss[i]:.4f}\t{m['eval_loss']:.4f}\t{m.get('bleu', 0):.2f}\t{m.get('rougeL', 0):.2f}\t{'; '.join(m.get('samples', []))}\n")

# ✅ Plot
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

print(f"\n{Fore.GREEN}Training complete! Results saved to {log_dir}{Style.RESET_ALL}")