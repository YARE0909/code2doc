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
import warnings
warnings.filterwarnings("ignore")

# Initialize colorama
from colorama import init
init()

# ✅ Mac GPU Setup
print("Setting up Mac GPU (MPS)...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"{Fore.GREEN}✓ MPS (Mac GPU) is available and will be used{Style.RESET_ALL}")
else:
    device = torch.device("cpu")
    print(f"{Fore.YELLOW}⚠ MPS not available, using CPU{Style.RESET_ALL}")

# Enable optimizations for Mac
torch.backends.mps.allow_tf32 = True if hasattr(torch.backends.mps, 'allow_tf32') else None

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

nltk.download("punkt", quiet=True)

def compute_metrics(eval_pred):
    """Fixed compute_metrics function with proper error handling"""
    try:
        predictions, labels = eval_pred
        
        # Handle tuple predictions (when model returns loss + predictions)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Ensure we have proper numpy arrays
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
            
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        print(f"\nDEBUG: Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")
        print(f"DEBUG: Predictions dtype: {predictions.dtype}, Labels dtype: {labels.dtype}")
        
        # Decode predictions
        decoded_preds = []
        for pred in predictions:
            try:
                # Ensure prediction is valid token IDs
                pred = np.where(pred >= 0, pred, tokenizer.pad_token_id)
                pred = np.where(pred < tokenizer.vocab_size, pred, tokenizer.pad_token_id)
                decoded = tokenizer.decode(pred, skip_special_tokens=True)
                decoded_preds.append(decoded.strip() if decoded.strip() else "empty")
            except Exception as e:
                print(f"Error decoding prediction: {e}")
                decoded_preds.append("error")
        
        # Decode labels
        decoded_labels = []
        for label in labels:
            try:
                # Replace -100 with pad token
                label = np.where(label != -100, label, tokenizer.pad_token_id)
                label = np.where(label >= 0, label, tokenizer.pad_token_id)
                label = np.where(label < tokenizer.vocab_size, label, tokenizer.pad_token_id)
                decoded = tokenizer.decode(label, skip_special_tokens=True)
                decoded_labels.append(decoded.strip() if decoded.strip() else "empty")
            except Exception as e:
                print(f"Error decoding label: {e}")
                decoded_labels.append("error")
        
        # Filter out empty/error predictions and labels
        valid_pairs = [(p, l) for p, l in zip(decoded_preds, decoded_labels) 
                      if p not in ["empty", "error"] and l not in ["empty", "error"]]
        
        if not valid_pairs:
            print("No valid prediction-label pairs found!")
            return {"bleu": 0.0, "rougeL": 0.0}
        
        filtered_preds, filtered_labels = zip(*valid_pairs)
        
        print(f"DEBUG: Valid pairs: {len(valid_pairs)}")
        print(f"DEBUG: Sample prediction: '{filtered_preds[0]}'")
        print(f"DEBUG: Sample label: '{filtered_labels[0]}'")
        
        # Initialize metrics
        metrics = {}
        
        # Compute BLEU
        try:
            bleu_result = bleu.compute(
                predictions=list(filtered_preds),
                references=[[label] for label in filtered_labels],
                smooth=True
            )
            metrics["bleu"] = bleu_result["bleu"] if bleu_result["bleu"] is not None else 0.0
            print(f"DEBUG: BLEU computed: {metrics['bleu']}")
        except Exception as e:
            print(f"BLEU computation error: {e}")
            metrics["bleu"] = 0.0
        
        # Compute ROUGE
        try:
            rouge_result = rouge.compute(
                predictions=list(filtered_preds),
                references=list(filtered_labels),
                use_stemmer=True
            )
            metrics["rougeL"] = rouge_result["rougeL"] * 100 if rouge_result["rougeL"] is not None else 0.0
            print(f"DEBUG: ROUGE-L computed: {metrics['rougeL']}")
        except Exception as e:
            print(f"ROUGE computation error: {e}")
            metrics["rougeL"] = 0.0
        
        # Print sample predictions for debugging
        print(f"\n{Fore.BLUE}Sample predictions vs references:{Style.RESET_ALL}")
        for i in range(min(3, len(filtered_preds))):
            print(f"{Fore.YELLOW}Pred:{Style.RESET_ALL} {filtered_preds[i]}")
            print(f"{Fore.CYAN}Ref:{Style.RESET_ALL} {filtered_labels[i]}")
            print("-" * 40)
        
        return metrics
    
    except Exception as e:
        print(f"Overall metric computation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"bleu": 0.0, "rougeL": 0.0}

# ✅ Training Args (adjusted for Mac)
training_args = Seq2SeqTrainingArguments(
    output_dir="./code2doc_model",
    eval_strategy="epoch",            # Evaluate at end of each epoch
    save_strategy="epoch",
    save_steps=1000,
    learning_rate=3e-4,
    warmup_steps=100,                 # Reduced warmup for single epoch
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,               # Single epoch as requested
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=2,
    logging_dir="./logs",
    logging_steps=50,                 # More frequent logging
    save_total_limit=2,
    report_to="none",
    fp16=False,
    dataloader_num_workers=0,
    eval_accumulation_steps=4,
    group_by_length=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
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

# Move model to device
model = model.to(device)
print(f"Model moved to device: {device}")

def run_test():
    """Test function with proper device handling"""
    try:
        input_text = "summarize: function test() { return 1; }"
        label_text = "Test function that returns 1"
        
        print(f"\n{'='*50}\nTest Debugging\n{'='*50}")
        print(f"Input text: {input_text}")
        print(f"Label text: {label_text}")
        
        inputs = tokenizer(input_text, return_tensors="pt")
        labels = tokenizer(label_text, return_tensors="pt").input_ids
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        print(f"Device: {device}")
        print("Tokenized input IDs:", inputs["input_ids"])
        print("Tokenized labels:", labels)
        
        with torch.no_grad():
            test_preds = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=2,
                early_stopping=True,
                do_sample=False
            )
        
        print("Raw predictions tensor:", test_preds)
        
        decoded_input = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        decoded_pred = tokenizer.decode(test_preds[0], skip_special_tokens=True)
        decoded_label = tokenizer.decode(labels[0], skip_special_tokens=True)
        
        print("Decoded input:", decoded_input)
        print("Decoded prediction:", decoded_pred)
        print("Decoded label:", decoded_label)
        
        # Test metrics computation
        test_preds_cpu = test_preds.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        
        print(f"Test predictions shape: {test_preds_cpu.shape}")
        print(f"Test labels shape: {labels_cpu.shape}")
        
        metrics = compute_metrics((test_preds_cpu, labels_cpu))
        print("Test metrics:", metrics)
        return metrics
    
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"bleu": 0.0, "rougeL": 0.0}

# Run test before training
print(f"{Fore.GREEN}Running initial test...{Style.RESET_ALL}")
test_metrics = run_test()

# ✅ Train
print(f"{Fore.GREEN}Starting training for 1 epoch...{Style.RESET_ALL}")
if device.type == "mps":
    # Clear MPS cache before training
    torch.mps.empty_cache()

train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
trainer.save_model("./code2doc_model_final")
tokenizer.save_pretrained("./code2doc_model_final")

# ✅ Display final results
print(f"\n{Fore.GREEN}{'='*60}")
print(f"TRAINING COMPLETE - FINAL RESULTS")
print(f"{'='*60}{Style.RESET_ALL}")

# Get the final evaluation metrics
final_eval_result = trainer.evaluate()
print(f"{Fore.CYAN}Final Validation Loss:{Style.RESET_ALL} {final_eval_result.get('eval_loss', 'N/A'):.4f}")
print(f"{Fore.YELLOW}Final BLEU Score:{Style.RESET_ALL} {final_eval_result.get('eval_bleu', 'N/A'):.4f}")
print(f"{Fore.YELLOW}Final ROUGE-L Score:{Style.RESET_ALL} {final_eval_result.get('eval_rougeL', 'N/A'):.4f}")

# Write final results to log
with open(log_file, "a") as f:
    f.write(f"FINAL\t{train_result.training_loss:.4f}\t{final_eval_result.get('eval_loss', 0):.4f}\t{final_eval_result.get('eval_bleu', 0):.4f}\t{final_eval_result.get('eval_rougeL', 0):.4f}\t\n")

# Remove the complex log processing and plotting since we only have 1 epoch
print(f"\n{Fore.GREEN}Training complete! Results saved to {log_dir}{Style.RESET_ALL}")
print(f"{Fore.BLUE}Final model saved to: ./code2doc_model_final{Style.RESET_ALL}")

# Simple final test
print(f"\n{Fore.GREEN}Running final test...{Style.RESET_ALL}")
final_test_metrics = run_test()