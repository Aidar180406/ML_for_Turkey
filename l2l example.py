import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Допустим, у нас есть 7 видео, каждое описано вектором из 4 признаков
# (в реальности это эмбеддинги от нейросети или статистика просмотров)
items = {
    "v1": "Аэрогриль: обзор и тесты",
    "v2": "Духовой шкаф: сравнение моделей",
    "v3": "Мультиварка: рецепты и режимы",
    "v4": "Футбольный матч: обзор тура",
    "v5": "Хоккей: лучшие голы сезона",
    "v6": "Как выбрать аэрогриль: гид для новичков",
    "v7": "Обзор кофемашины"
}

# Векторы (эмбеддинги) — упрощённо по темам: [кухня, спорт, техника, напитки]
embeddings = np.array([
    [0.9, 0.1, 0.8, 0.1],  # v1 — аэрогриль (кухня + техника)
    [0.8, 0.1, 0.7, 0.2],  # v2 — духовой шкаф (кухня + техника)
    [0.9, 0.0, 0.6, 0.1],  # v3 — мультиварка
    [0.1, 0.9, 0.1, 0.0],  # v4 — футбол (спорт)
    [0.0, 0.8, 0.2, 0.1],  # v5 — хоккей (спорт)
    [0.8, 0.1, 0.8, 0.1],  # v6 — гид по аэрогрилю (похоже на v1)
    [0.2, 0.1, 0.3, 0.8]   # v7 — кофемашина (напитки + техника)
])

# 2. Косинусная матрица похожести (все со всеми)
similarity_matrix = cosine_similarity(embeddings)

def get_i2i_recommendations(item_id, top_k=5):
    idx = list(items.keys()).index(item_id)
    # Берём строку с похожестью этого объекта ко всем остальным
    sims = similarity_matrix[idx]
    # Сортируем по убыванию, исключая сам объект
    top_indices = np.argsort(sims)[::-1]
    recs = []
    for i in top_indices:
        if list(items.keys())[i] == item_id:
            continue
        recs.append((list(items.keys())[i], float(sims[i])))
        if len(recs) == top_k:
            break
    return recs

# Пример: пользователь посмотрел v1 (аэрогриль)
user_watched = "v1"
recommendations = get_i2i_recommendations(user_watched, top_k=5)

print(f"Пользователь посмотрел: {items[user_watched]}")
print("Рекомендации (Item-to-Item):")
for rec_id, score in recommendations:
    print(f"- {items[rec_id]} (схожесть: {score:.3f})")
