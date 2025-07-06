import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

# Datasets & splits to analyze
DATASETS = [
    ("code_search_net", "javascript"),
    ("code_search_net", "python"),
    ("code_x_glue_ct_code_to_text", "javascript"),
    ("code_x_glue_ct_code_to_text", "python"),
]

OUTPUT_DIR = "eda_lengths_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pick_fields(columns):
    code_opts = ["code", "func_code_string", "whole_func_string"]
    doc_opts  = ["docstring", "func_documentation_string"]
    code_field = next(c for c in code_opts if c in columns)
    doc_field  = next(c for c in doc_opts  if c in columns)
    return code_field, doc_field

for ds_name, subset in DATASETS:
    print(f"\n▶ Processing {ds_name}/{subset}")

    # 1) Open a streaming iterator and pull one example to detect fields
    stream_peek = load_dataset(ds_name, subset, split="train", streaming=True)
    first_example = next(iter(stream_peek))
    code_field, doc_field = pick_fields(first_example.keys())
    print(f"   • Detected fields → code: '{code_field}', doc: '{doc_field}'")

    # 2) Re‑open the stream for real processing
    ds_stream = load_dataset(ds_name, subset, split="train", streaming=True)

    # 3) Collect token‐count statistics up to N examples
    code_lens, doc_lens = [], []
    N = 20_000  # you can reduce this if you want even lighter RAM usage
    for i, ex in enumerate(ds_stream):
        code_lens.append(len(ex[code_field].split()))
        doc_lens.append(len(ex[doc_field].split()))
        if i + 1 >= N:
            break
    print(f"   • Collected {len(code_lens)} examples")

    # 4) Convert to NumPy arrays and compute numeric summary
    code_arr = np.array(code_lens, dtype=int)
    doc_arr  = np.array(doc_lens, dtype=int)
    summary = {
        "count": len(code_arr),
        "code_min": int(code_arr.min()),
        "code_25%": int(np.percentile(code_arr, 25)),
        "code_median": int(np.median(code_arr)),
        "code_mean": float(code_arr.mean()),
        "code_75%": int(np.percentile(code_arr, 75)),
        "code_max": int(code_arr.max()),
        "doc_min": int(doc_arr.min()),
        "doc_25%": int(np.percentile(doc_arr, 25)),
        "doc_median": int(np.median(doc_arr)),
        "doc_mean": float(doc_arr.mean()),
        "doc_75%": int(np.percentile(doc_arr, 75)),
        "doc_max": int(doc_arr.max()),
    }

    # 5) Write out the summary TXT
    sum_path = os.path.join(OUTPUT_DIR, f"{ds_name}_{subset}_summary.txt")
    with open(sum_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"   ✔ Summary → {sum_path}")

    # 6) Plot and save histograms for code & doc lengths
    for arr, label in [(code_arr, "code"), (doc_arr, "doc")]:
        plt.figure()
        plt.hist(arr, bins=50)
        plt.title(f"{ds_name}/{subset} {label.capitalize()} Lengths")
        plt.xlabel("Token Count")
        plt.ylabel("Frequency")
        out_png = os.path.join(OUTPUT_DIR, f"{ds_name}_{subset}_{label}_hist.png")
        plt.savefig(out_png)
        plt.close()
        print(f"   ✔ Plot → {out_png}")

print("\n✅ All done. Check the ‘eda_lengths_output’ folder for results.")  
