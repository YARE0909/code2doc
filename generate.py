import os, glob, torch
from datetime import datetime
from transformers import AutoTokenizer, T5ForConditionalGeneration

# 1) Load your trained model & tokenizer
MODEL_DIR = "./code2doc_model_final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_doc(code_snippet: str,
                 max_input_length: int = 512,
                 max_output_length: int = 128,
                 num_beams: int = 4):
    text = "summarize: " + code_snippet.strip()
    inputs = tokenizer(
        text,
        max_length=max_input_length,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    outs = model.generate(
        input_ids      = inputs.input_ids,
        attention_mask = inputs.attention_mask,
        max_length     = max_output_length,
        num_beams      = num_beams,
        early_stopping = True,
    )

    return tokenizer.decode(outs[0], skip_special_tokens=True).strip()

def get_lang_tag(filename: str):
    ext = os.path.splitext(filename)[-1]
    return {
        '.js': 'javascript',
        '.py': 'python',
        '.java': 'java',
        '.go': 'go',
    }.get(ext, '')

def batch_generate(input_dir: str, base_output: str = "./output/results"):
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # File extensions to support
    extensions = ["*.js", "*.py", "*.java", "*.go"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()

        if not code or len(code) > 50_000:
            continue

        doc = generate_doc(code)
        
        # Output file setup
        base_name = os.path.basename(path)
        lang_tag = get_lang_tag(base_name)
        out_path = os.path.join(output_dir, base_name + ".docs.md")

        with open(out_path, "w", encoding="utf-8") as out:
            out.write(f"# Documentation for `{base_name}`\n\n")
            out.write(f"```{lang_tag}\n{code}\n```\n\n")
            out.write("## Generated Documentation\n\n")
            out.write(doc + "\n")

        print(f"📝 {path} → {out_path}")

if __name__ == "__main__":
    project_dir = "./examples"
    batch_generate(project_dir)
