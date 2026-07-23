import json
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Groq()

model = SentenceTransformer("all-MiniLM-L6-v2")

def distance(v1, v2):
    diff = v1 - v2
    squared = diff ** 2
    total = squared.sum()
    return total ** 0.5

docs = {
    "a": "Партнёр получает 30% от суммы платежей приведённого клиента. Выплаты раз в месяц, минимум для вывода — 50 долларов, на USDT или банковскую карту.",
    "b": "Реферальная ссылка живёт 60 дней. Если клиент перешёл по ссылке и оплатил в течение этого срока — засчитывается партнёру, даже если вернулся позже.",
    "c": "Партнёрка открыта для всех стран, кроме США и Канады. Один человек может завести только один партнёрский аккаунт, мультиаккаунты банятся.",
}

base = {}
for name in docs:
    base[name] = model.encode(docs[name])

def search_docs(query):
    query_vector = model.encode(query)
    distances = {}

    for name in base:
        distances[name] = distance(query_vector, base[name])

    ближайший = min(distances, key=distances.get)
    return docs[ближайший]



def multiply(a, b):
    return a * b


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

if os.path.exists("history.json"):
    with open("history.json", "r", encoding="utf-8") as f:
        messages = json.load(f)
else:
    messages = []

while True:
    user_input = input("You: ")

    if user_input == "exit":
        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        break

    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": message.tool_calls[0].id,
                            "type": "function",
                            "function": {
                                "name": message.tool_calls[0].function.name,
                                "arguments": message.tool_calls[0].function.arguments,
                            },
                        }
                    ],
                }
            )

            tool_name = message.tool_calls[0].function.name
            tool_args = json.loads(message.tool_calls[0].function.arguments)

            if tool_name == "search_docs":
                result = search_docs(tool_args["query"])
            elif tool_name == "multiply":
                result = multiply(tool_args["a"], tool_args["b"])
            

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_calls[0].id,
                    "content": str(result),
                }
            )

            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )

            final_message = final_response.choices[0].message
            print("Bot:", final_message.content)
            messages.append({"role": "assistant", "content": final_message.content})

        else:
            print("Bot:", message.content)
            messages.append({"role": "assistant", "content": message.content})
    except Exception as e:
        print("Bot:", "Произошла ошибка:", e)