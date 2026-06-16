# Some useful libraries, feel free to import any others you need.
import os
import torch
import time
import json
import jailbreakbench as jbb

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from gcg.algorithm import GCGConfig, run

from config import NUM_SAMPLES, MODEL_NAME, RESULT_DIR, SELECTED_INDICES

import os


model_id = MODEL_NAME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the model and tokenizer
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

# Fix pad token for generation and batching.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

if hasattr(model, "config"):
    model.config.pad_token_id = tokenizer.pad_token_id

if hasattr(model, "generation_config"):
    model.generation_config.pad_token_id = tokenizer.pad_token_id


# 2. Load the harmful queries and target responses

dataset = jbb.read_dataset()

if SELECTED_INDICES is None:
    selected_indices = list(range(NUM_SAMPLES))
else:
    selected_indices = SELECTED_INDICES

queries = [dataset.goals[i] for i in selected_indices]
targets = [dataset.targets[i] for i in selected_indices]


# 3. Run GCG

config = GCGConfig(
    num_steps=250,           # You can try to adjust these to lower the runtime but be sure it doesn't hinder the attack success.
    search_width=512,
    topk=256,
    verbosity="WARNING",     # Set to "INFO" for more detailed output
    use_prefix_cache=False
)

suffixes = []
losses = []
start_time = time.time()
for data_index, query, target in tqdm(zip(selected_indices, queries, targets), total=len(queries), desc="Running GCG", unit="query"):
    result = run(model, tokenizer, query, target, config)
    suffixes.append({
        "index": data_index,
        "query": query,
        "target": target,
        "suffix": result.best_string,
        "loss": result.best_loss,
    })
    losses.append(result.best_loss)

print(f"Time taken: {time.time() - start_time} seconds")
print("Average loss: ", sum(losses) / len(losses))


# 4. Save the adversarial suffixes

os.makedirs(RESULT_DIR, exist_ok=True)

suffix_path = os.path.join(RESULT_DIR, "suffixes.json")

with open(suffix_path, "w", encoding="utf-8") as f:
    json.dump(suffixes, f, indent=4, ensure_ascii=False)

print(f"Suffixes saved to {suffix_path}")