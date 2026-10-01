import json

with open("evals/dataset.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        print(row["id"])