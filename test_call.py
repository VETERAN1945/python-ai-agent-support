from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def distance(v1, v2):
    diff = v1 - v2
    squares = diff ** 2
    total = squares.sum()
    return total ** 0.5


docs = {
    "a": "кошка спит на диване",
    "b": "кот дремлет на кровати",
    "c": "доллар вырос к евро",
}

# ─── ЭТАЖ 1: голый файл, бежит ОДИН раз при запуске ───
# МОМЕНТ 1 — индексация: все доки → векторы → на склад
base = {}
for name, text in docs.items():
    base[name] = model.encode(text)


# ─── ЭТАЖ 2: тело функции, спит до вызова, бежит НА КАЖДЫЙ вызов ───
def search_docs(вопрос):
    # МОМЕНТ 2 — поиск: вектор вопроса → distance до готовых векторов базы
    q_vec = model.encode(вопрос)          # дырка 1: что суём в encode здесь?

    distances = {}
    for name in base:
        distances[name] = distance(q_vec, base[name])   # дырка 2: откуда готовый вектор дока?

    ближайший = min(distances, key=distances.get)
              
    return docs[ближайший]


# ─── проба руками (потом это будет дёргать модель) ───
ответ = search_docs("животное отдыхает")
print("Нашёл документ:", ответ)