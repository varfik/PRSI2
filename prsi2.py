import os
import re
print(os.listdir())

filename = "Достоевский том 1-5.txt"

with open(filename, "r", encoding="utf-8") as file:
    text = file.read()

print(text[:1000])

text = text.lower()
text = re.sub(r"[^\w\s]", "", text)  # Убираем знаки препинания
text = re.sub(r"\d+", "", text)      # Убираем числа
text = re.sub(r"\s+", " ", text).strip()
print(text[:1000])

text = text.split()
tokens = text
print(len(text))
print(len(tokens))

# Создаём словарь "слово -> индекс"
word2idx = {word: idx for idx, word in enumerate(set(tokens))}

# Обратный словарь "индекс -> слово"
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)
print("Размер словаря:", vocab_size)

import numpy as np

L = 4  # Количество слов в контексте

# Создаём пары (контекст → целевое слово)
data = []
for i in range(L, len(tokens) - L):
    context = tokens[i-L:i]  # L слов до целевого
    target = tokens[i]       # Следующее слово
    data.append((context, target))

# Выводим первые 5 примеров
for i in range(30):
    print("Контекст:", data[i][0], "→ Целевое слово:", data[i][1])

import numpy as np

L = 4  # Количество слов в контексте
embedding_dims = [100, 500, 1000]  # Три варианта d
epochs = 3
learning_rate = 0.01

# Размер словаря
vocab_size = len(word2idx)

# Функция инициализации весов
def initialize_weights(vocab_size, d):
    W1 = np.random.randn(vocab_size, d) * 0.01  # Матрица эмбеддингов
    W2 = np.random.randn(d, vocab_size) * 0.01  # Выходная матрица
    return W1, W2

def one_hot_encoding(word, vocab_size):
    vector = np.zeros(vocab_size)
    vector[word2idx[word]] = 1
    return vector

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Для числовой стабильности
    return exp_x / exp_x.sum(axis=0)

def forward(context_words, W1, W2):
    # Усредняем one-hot векторы контекста
    x = np.mean(np.array([one_hot_encoding(word, vocab_size) for word in context_words]), axis=0)

    # Первый слой (входные эмбеддинги)
    h = np.dot(W1.T, x)

    # Второй слой (выходное распределение)
    u = np.dot(W2.T, h)
    y_pred = softmax(u)

    return x, h, y_pred

def cross_entropy_loss(y_pred, y_true):
    return -np.log(y_pred + 1e-9)  # Добавляем 1e-9 для избежания log(0)

def backward(x, h, y_pred, y_true, W1, W2, learning_rate):
    # Градиент ошибки
    y_true_one_hot = one_hot_encoding(y_true, vocab_size)
    error = y_pred - y_true_one_hot

    # Градиенты матриц W2 и W1
    dW2 = np.outer(h, error)
    dW1 = np.outer(x, np.dot(W2, error))

    # Обновление весов
    W2 -= learning_rate * dW2
    W1 -= learning_rate * dW1

for d in embedding_dims:
    print(f"\n🔵 Обучаем Word2Vec (d = {d})...\n")

    # Инициализируем веса
    W1, W2 = initialize_weights(vocab_size, d)

    for epoch in range(epochs):
        total_loss = 0

        for context, target in data:
            # Прямой проход
            x, h, y_pred = forward(context, W1, W2)

            # Вычисляем ошибку
            loss = cross_entropy_loss(y_pred, target)
            total_loss += loss

            # Обратное распространение
            #backward(x, h, y_pred, target, W1, W2, learning_rate)

        print(f"Эпоха {epoch+1}/{epochs}, Потери: {total_loss:.4f}")

test_sentence = ["высокий", "худой", "мужчина", "подошел"]
test_indices = [word2idx[word] for word in test_sentence]

for d in embedding_dims:
    print(f"\n🔹 d = {d}:")

    W1, W2 = initialize_weights(vocab_size, d)  # Загружаем обученные веса

    _, _, y_pred = forward(test_indices, W1, W2)

    predicted_word = idx2word[np.argmax(y_pred)]

    print(f"Предсказанное слово: {predicted_word}")