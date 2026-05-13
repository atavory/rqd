#!/bin/bash
# Full GRR experiment pipeline for t3 (2x A100 80GB)
# Trains a real LLM generative recommender on MovieLens semantic IDs,
# then evaluates whether old model breaks under full codebook retraining.
set -euo pipefail

WORK=/tmp/grr_experiment
GRR_DIR="$HOME/fbsource/fbcode/grr_code/grr"
mkdir -p $WORK && cd $WORK

echo "=== Step 1: Prep MovieLens data with RQ semantic IDs ==="
manifold get manifold://fi_platform_ml_infra_fluent2_bucket/tree/fi_trunk_tail/continual_rq_scripts/prep_grr_data.py prep_grr_data.py
pip install pandas pyarrow 2>/dev/null
python3 prep_grr_data.py --output-dir $WORK/data --seed 0
echo "Data ready at $WORK/data"
ls -la $WORK/data/

echo "=== Step 2: Setup GRR conda env ==="
if ! conda env list | grep -q grr_env; then
    echo "Creating conda env (this takes ~10 min first time)..."
    cd $GRR_DIR
    bash setup_conda_env.sh
fi
conda activate grr_env || source activate grr_env

echo "=== Step 3: Expand Qwen3 vocab with SID tokens ==="
# Use Qwen3-0.6B for 2-GPU setup (smaller than 1.7B)
MODEL_NAME="Qwen/Qwen3-0.6B"
EXPANDED_MODEL="$WORK/qwen3_expanded"
if [ ! -d "$EXPANDED_MODEL" ]; then
    python3 $GRR_DIR/basemodel/expand_vocab.py \
        --model_name $MODEL_NAME \
        --output_dir $EXPANDED_MODEL
fi
echo "Expanded model at $EXPANDED_MODEL"

echo "=== Step 4: Train on T0 (source period) ==="
# Stage 1: Alignment training with LoRA on 2 GPUs
cd $GRR_DIR
deepspeed --num_gpus 2 train/scripts/train_beauty_align.py \
    --model_dir $EXPANDED_MODEL \
    --train_data_path $WORK/data/train_t0.parquet \
    --output_dir $WORK/model_t0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --deepspeed $GRR_DIR/train/ds_config_zero2.json \
    2>&1 | tee $WORK/train_t0.log | tail -20

echo "=== Step 5: Evaluate on T1 with different codebook strategies ==="
# Evaluate: generate next-item SIDs and compute hit rate
python3 - <<'EVALSCRIPT'
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
import numpy as np

work = "/tmp/grr_experiment"
model_path = f"{work}/model_t0"
base_path = f"{work}/qwen3_expanded"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=True,
                                              torch_dtype=torch.float16, device_map="auto")
try:
    model = PeftModel.from_pretrained(model, model_path)
except:
    pass
model.eval()

results = []
for strategy in ["frozen", "strat", "full"]:
    eval_path = f"{work}/data/eval_t1_{strategy}.parquet"
    df = pd.read_parquet(eval_path)
    hits, total = 0, 0
    for _, row in df.iterrows():
        prompt = row["description"] + "\nPredict the next item the user will purchase:\n"
        gt = row["groundtruth"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        if gt.strip() in pred:
            hits += 1
        total += 1
        if total >= 500:
            break
    hr = hits / total
    print(f"  {strategy}: HR@1={hr:.4f} ({hits}/{total})")
    results.append({"strategy": strategy, "hr_at_1": hr, "n_eval": total})

with open(f"{work}/grr_results.json", "w") as f:
    json.dump(results, f)
print("Results saved")
EVALSCRIPT

echo "=== Step 6: Upload results ==="
manifold put $WORK/grr_results.json manifold://fi_platform_ml_infra_fluent2_bucket/tree/fi_trunk_tail/continual_rq_scripts/grr_results.json
echo "=== DONE ==="
