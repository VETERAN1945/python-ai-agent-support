import json
import time                      # new
import httpx                     # new

run_id = str(int(time.time()))   # new

with open("evals/dataset.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        resp = httpx.post(                                   # new
            "http://127.0.0.1:8000/chat",                    # new
            json={"user_id": run_id + "_" + row["id"],       # new
                  "message": row["question"]},                   # new
            timeout=60,                                      # new
        )                                                    # new
        data = resp.json()                                   # new
        hit = row["expected_doc"] == data["retrieved_doc"]
        print(row["id"], hit, data["reply"][:60])