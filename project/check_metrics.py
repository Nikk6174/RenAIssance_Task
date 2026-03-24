import json

data = json.load(open("models/t5/checkpoint-770/trainer_state.json"))
logs = data["log_history"]

train = [l for l in logs if "loss" in l and "eval_loss" not in l]
evals = [l for l in logs if "eval_loss" in l]

print("TRAIN LOSSES:")
for l in train:
    print(f"  step {l['step']}: {l['loss']:.4f}")

print("\nEVAL LOSSES:")
for l in evals:
    cer = l.get("eval_cer", "N/A")
    print(f"  step {l['step']}: eval_loss={l['eval_loss']:.4f}, eval_cer={cer}")

print(f"\nBest metric: {data.get('best_metric', 'N/A')}")
