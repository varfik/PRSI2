import os
import re
import numpy as np
from tqdm import tqdm  # Для красивого прогресс-бара
import time  # Для измерения времени

print(os.listdir())

filename = "Достоевский том 1-5.txt"

# Чтение файла с обработкой ошибок
try:
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
except FileNotFoundError:
    print(f"Файл {filename} не найден!")
    exit()

print("Первые 1000 символов текста:")
print(text[:1000])

# Предварительная обработка текста
text = text.lower()
text = re.sub(r"[^\w\s]", "", text)  # Убираем знаки препинания
text = re.sub(r"\d+", "", text)      # Убираем числа
text = re.sub(r"\s+", " ", text).strip()
print("\nТекст после обработки (первые 1000 символов):")
print(text[:1000])

# Токенизация
text = text.split()
tokens = text
print(f"\nОбщее количество токенов: {len(tokens)}")

# Создание словаря
word2idx = {word: idx for idx, word in enumerate(set(tokens))}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)
print(f"Размер словаря: {vocab_size}")

# Параметры модели
L = 4  # Количество слов в контексте
embedding_dims = [100, 500, 1000]  # Размерности эмбеддингов
epochs = 3
learning_rate = 0.01
batch_size = 128  # Размер батча для ускорения обучения

# Создание обучающих данных
print("\nСоздание обучающих данных...")
data = []
for i in range(L, len(tokens) - L):
    context = tokens[i-L:i]
    target = tokens[i]
    data.append((context, target))

print(f"Создано {len(data)} обучающих примеров")
print("\nПримеры данных:")
for i in range(5):
    print(f"{i+1}. Контекст: {data[i][0]} → Целевое слово: {data[i][1]}")

# Функции модели
def initialize_weights(vocab_size, d):
    W1 = np.random.randn(vocab_size, d) * 0.01
    W2 = np.random.randn(d, vocab_size) * 0.01
    return W1, W2

def one_hot_encoding(word, vocab_size):
    vector = np.zeros(vocab_size)
    vector[word2idx[word]] = 1
    return vector

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def forward(context_words, W1, W2):
    x = np.mean(np.array([one_hot_encoding(word, vocab_size) for word in context_words]), axis=0)
    h = np.dot(W1.T, x)
    u = np.dot(W2.T, h)
    y_pred = softmax(u)
    return x, h, y_pred

def cross_entropy_loss(y_pred, y_true):
    return -np.log(y_pred[y_true] + 1e-9)

def backward(x, h, y_pred, y_true, W1, W2, learning_rate):
    y_true_one_hot = one_hot_encoding(y_true, vocab_size)
    error = y_pred - y_true_one_hot
    dW2 = np.outer(h, error)
    dW1 = np.outer(x, np.dot(W2, error))
    return dW1, dW2

# Обучение модели
for d in embedding_dims:
    print(f"\n{'='*50}")
    print(f"🔵 Начинаем обучение Word2Vec (размерность эмбеддинга d = {d})")
    print(f"Размер словаря: {vocab_size}")
    print(f"Количество обучающих примеров: {len(data)}")
    print(f"Количество эпох: {epochs}")
    print(f"Скорость обучения: {learning_rate}")
    print(f"Размер батча: {batch_size}")
    print(f"Дата начала: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")
    
    W1, W2 = initialize_weights(vocab_size, d)
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_examples = 0
        
        # Создаем прогресс-бар для эпохи
        with tqdm(data, desc=f"Эпоха {epoch+1}/{epochs}", unit="пример") as pbar:
            for i, (context, target) in enumerate(pbar):
                # Прямой проход
                x, h, y_pred = forward(context, W1, W2)
                
                # Вычисление потерь и точности
                loss = cross_entropy_loss(y_pred, target)
                epoch_loss += loss
                
                predicted_idx = np.argmax(y_pred)
                if predicted_idx == word2idx[target]:
                    correct_predictions += 1
                total_examples += 1
                
                # Обратное распространение (обновление весов)
                dW1, dW2 = backward(x, h, y_pred, target, W1, W2, learning_rate)
                W1 -= learning_rate * dW1
                W2 -= learning_rate * dW2
                
                # Обновляем прогресс-бар каждые 100 примеров
                if i % 100 == 0:
                    pbar.set_postfix({
                        'Потери': f"{epoch_loss/(i+1):.4f}",
                        'Точность': f"{correct_predictions/(total_examples+1e-9):.2%}",
                    })
        
        # Статистика после эпохи
        avg_loss = epoch_loss / len(data)
        accuracy = correct_predictions / len(data)
        print(f"\nРезультаты эпохи {epoch+1}:")
        print(f"Средние потери: {avg_loss:.4f}")
        print(f"Точность: {accuracy:.2%}")
        print(f"Время эпохи: {time.time() - start_time:.2f} сек\n")
    
    total_time = time.time() - start_time
    print(f"Обучение завершено! Общее время: {total_time:.2f} сек")
    print(f"Среднее время на эпоху: {total_time/epochs:.2f} сек")

# Тестирование модели
test_phrases = [
    ["высокий", "худой", "мужчина", "подошел"],
    ["князь", "сказал", "что", "он"],
    ["она", "посмотрела", "на", "него"]
]

print("\nТестирование модели:")
for d in embedding_dims:
    print(f"\n🔹 Размерность эмбеддинга d = {d}:")
    W1, W2 = initialize_weights(vocab_size, d)
    
    for phrase in test_phrases:
        try:
            _, _, y_pred = forward(phrase, W1, W2)
            predicted_word = idx2word[np.argmax(y_pred)]
            print(f"Контекст: {phrase} → Предсказание: '{predicted_word}'")
        except KeyError as e:
            print(f"Ошибка: слово {e} отсутствует в словаре")
