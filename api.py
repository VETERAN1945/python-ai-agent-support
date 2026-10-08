from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import json

load_dotenv()
app = FastAPI()
client = Groq()
sessions = {}
LLM = "openai/gpt-oss-120b"

docs = {
    "a": "Партнёр получает 30% от суммы платежей приведённого клиента. Выплаты раз в месяц, минимум для вывода — 50 долларов, на USDT или банковскую карту.",
    "b": "Реферальная ссылка живёт 60 дней. Если клиент перешёл по ссылке и оплатил в течение этого срока — засчитывается партнёру, даже если вернулся позже.",
    "c": "Партнёрка открыта для всех стран, кроме США и Канады. Один человек может завести только один партнёрский аккаунт, мультиаккаунты банятся.",
}

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

base = {}
for key, text in docs.items():
    base[key] = model.encode(text)
print("ФОРМА ВЕКТОРА:", base["a"].shape)


def multiply(a, b):
    return a * b


def search_docs(query):
    q_vec = model.encode(query)
    best_key = None
    best_dist = 999
    for key, doc_vec in base.items():
        dist = np.linalg.norm(q_vec - doc_vec)
        print("  дистанция", key, dist)
        if dist < best_dist:
            best_dist = dist
            best_key = key
    print("НАЙДЕН ДОК:", best_key)
    return docs[best_key], best_key


tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiplies two numbers and returns the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Searches the knowledge base for information about the affiliate program. Use this when a user asks about money, commissions, payouts, выплаты, сколько платят, комиссия, вознаграждение",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to search for"}
                },
                "required": ["query"]
            }
        }
    }
]


class Question(BaseModel):
    user_id: str
    message: str


@app.post("/chat")
def chat(q: Question):
    if q.user_id not in sessions:
        sessions[q.user_id] = []
    history = sessions[q.user_id]
    history.append({"role": "user", "content": q.message})
    snapshot = len(history)
    retrieved_doc = None

    try:
        response = client.chat.completions.create(
            model=LLM,
            messages=history,
            tools=tools,
        )
        message = response.choices[0].message

        if message.tool_calls:
            history.append(message)
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            tool_name = tool_call.function.name
            print("ИНСТРУМЕНТ:", tool_name, args)

            if tool_name == "multiply":
                result = multiply(**args)
            else:
                result, retrieved_doc = search_docs(q.message)

            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
            second = client.chat.completions.create(
                model=LLM,
                messages=history,
            )
            reply = second.choices[0].message.content
        else:
            reply = message.content

    except Exception as e:
        print(e)
        del history[snapshot:]
        reply = "Произошла ошибка сервера"

    history.append({"role": "assistant", "content": reply})
    print("ПОЛКА:", len(history))
    return {"reply": reply, "retrieved_doc": retrieved_doc}