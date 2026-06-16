import os
import json
import torch
import pandas as pd
from tqdm.notebook import tqdm
from config import NUM_SAMPLES, MODEL_NAME, RESULT_DIR, SELECTED_INDICES
import jailbreakbench as jbb
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = MODEL_NAME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
)

model.to(device)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

if hasattr(model, "config"):
    model.config.pad_token_id = tokenizer.pad_token_id

if hasattr(model, "generation_config"):
    model.generation_config.pad_token_id = tokenizer.pad_token_id

# ===== 1. 读取 suffixes =====

suffix_path = os.path.join(RESULT_DIR, "suffixes.json")

with open(suffix_path, "r", encoding="utf-8") as f:
    suffixes = json.load(f)

suffix_by_index = {
    item["index"]: item["suffix"]
    for item in suffixes
}

print(f"Loaded {len(suffixes)} suffixes from {suffix_path}")


# ===== 2. 读取对应 queries / targets =====

dataset = jbb.read_dataset()

if SELECTED_INDICES is None:
    selected_indices = list(range(NUM_SAMPLES))
else:
    selected_indices = SELECTED_INDICES

queries = [dataset.goals[i] for i in selected_indices]
targets = [dataset.targets[i] for i in selected_indices]


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

for i, (data_index, query, target) in enumerate(zip(selected_indices, queries, targets)):
    suffix = suffix_by_index[data_index]
    attacked_prompt = query + " " + suffix

    original_output = generate_response(
        query,
        max_new_tokens=128,
        use_chat_template=True,   # 如果不是 chat model，可以改成 False
    )

    attacked_output = generate_response(
        attacked_prompt,
        max_new_tokens=128,
        use_chat_template=True,   # 如果不是 chat model，可以改成 False
    )

    rows.append({
        "idx": i,
        "dataset_index": data_index,
        "query": query,
        "target": target,
        "suffix": suffix,
        "original_prompt": query,
        "attacked_prompt": attacked_prompt,
        "original_output": original_output,
        "attacked_output": attacked_output,
    })


# ===== 6. 保存结果 =====

df = pd.DataFrame(rows)

out_dir = RESULT_DIR
os.makedirs(out_dir, exist_ok=True)

csv_path = f"{out_dir}/comparison_outputs.csv"
json_path = f"{out_dir}/comparison_outputs.json"

df.to_csv(csv_path, index=False)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=4, ensure_ascii=False)

print(f"Saved CSV to: {csv_path}")
print(f"Saved JSON to: {json_path}")

df.head()
