import os
import json
import torch
import pandas as pd
from tqdm.notebook import tqdm
from config import MODEL_NAME
import jailbreakbench as jbb
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = MODEL_NAME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#TO DO
model = ...
tokenizer = ...

# ===== 1. 读取 suffixes =====

suffix_path = f"results/{MODEL_NAME}/suffixes.json"

with open(suffix_path, "r", encoding="utf-8") as f:
    suffixes = json.load(f)

NUM_SAMPLES = len(suffixes)

print(f"Loaded {NUM_SAMPLES} suffixes from {suffix_path}")


# ===== 2. 读取对应 queries / targets =====

dataset = jbb.read_dataset()

queries = dataset.goals[:NUM_SAMPLES]
targets = dataset.targets[:NUM_SAMPLES]


# ===== 3. 修复 pad_token，避免 generate 报错 =====

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

if hasattr(model, "config"):
    model.config.pad_token_id = tokenizer.pad_token_id

if hasattr(model, "generation_config"):
    model.generation_config.pad_token_id = tokenizer.pad_token_id


# ===== 4. 生成函数 =====

def generate_response(prompt, max_new_tokens=128, use_chat_template=False):
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_len:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ===== 5. 生成原始输出 vs 加 suffix 输出 =====

rows = []

for i, (query, target, suffix) in enumerate(zip(queries, targets, suffixes)):
    attacked_prompt = query + " " + suffix

    original_output = generate_response(
        query,
        max_new_tokens=128,
        use_chat_template=False,   # 如果不是 chat model，可以改成 False
    )

    attacked_output = generate_response(
        attacked_prompt,
        max_new_tokens=128,
        use_chat_template=False,   # 如果不是 chat model，可以改成 False
    )

    rows.append({
        "idx": i,
        "query": query,
        "target": target,
        "suffix": suffix,
        "attacked_prompt": attacked_prompt,
        "original_output": original_output,
        "attacked_output": attacked_output,
    })


# ===== 6. 保存结果 =====

df = pd.DataFrame(rows)

out_dir = f"results/{MODEL_NAME}"
os.makedirs(out_dir, exist_ok=True)

csv_path = f"{out_dir}/comparison_outputs.csv"
json_path = f"{out_dir}/comparison_outputs.json"

df.to_csv(csv_path, index=False)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=4, ensure_ascii=False)

print(f"Saved CSV to: {csv_path}")
print(f"Saved JSON to: {json_path}")

df.head()
