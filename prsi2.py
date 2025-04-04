import os
import re
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt

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
# Словарь для подсчета частоты слов
word_freq = defaultdict(int)
for word in text:
    word_freq[word] += 1

words = [word for word, freq in word_freq.items() if (len(word)>3) and (freq >= 6)]
print(len(text))
print(len(words))

# Словарь "слово - индекс"
word2idx = {word: idx for idx, word in enumerate(words)}

# Словарь "индекс - слово"
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)
print("Размер словаря:", vocab_size)

# Параметры модели
L = 4  # Количество слов в контексте
embedding_dims = [100, 500, 1000]  # Размерности эмбеддингов
epochs = 1
learning_rate = 0.1
batch_size = 128  # Размер батча для ускорения обучения

# Создание обучающих данных
def generate_cbow_data(text, word2idx, L=2):
    X, y = [], []
    for i in range(L, len(text) - L):
        context = [
            word2idx[text[j]] 
            for j in range(i - L, i + L + 1) 
            if j != i
        ]
        target = word2idx[text[i]]
        X.append(context)
        y.append(target)
    return np.array(X), np.array(y)

def plot_training_progress(epoch, total_epochs, loss, stage="Word2Vec"):
    progress = (epoch + 1) / total_epochs * 100
    plt.figure(figsize=(10, 2))
    plt.barh(0, progress, color='skyblue')
    plt.text(progress/2, 0, 
             f"{stage} | Эпоха {epoch+1}/{total_epochs} | Loss: {loss:.4f}", 
             va='center', ha='center', color='black', fontsize=10)
    plt.xlim(0, 100)
    plt.axis('off')
    plt.show()

L = 4
X_cbow, y_cbow = generate_cbow_data(words, word2idx, L)
print("Пример данных CBOW:", X_cbow[0], "→", y_cbow[0])
class Word2Vec:
    def __init__(self, vocab_size, d):
        self.W1 = np.random.randn(vocab_size, d) * 0.01  # Эмбеддинги
        self.W2 = np.random.randn(d, vocab_size) * 0.01  # Выходной слой

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        # Усредняем эмбеддинги контекстных слов
        hidden = np.mean(self.W1[X], axis=1)  # (batch_size, d)
        output = np.dot(hidden, self.W2)      # (batch_size, vocab_size)
        return self.softmax(output)

    def train(self, X_train, y_train, epochs=5, learning_rate=0.01):
        for epoch in range(epochs):
            loss = 0
            for i in tqdm(range(0, len(X_train), 
                          desc=f"Обучение (d={self.W1.shape[1]})",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"):
            for X, y in zip(X_train, y_train):
                # Forward pass
                hidden = np.mean(self.W1[X], axis=0)  # (d,)
                output = np.dot(hidden, self.W2)       # (vocab_size,)
                probs = self.softmax(output[np.newaxis, :])[0]

                # Ошибка (cross-entropy)
                loss += -np.log(probs[y])

                # Backpropagation
                grad_output = probs
                grad_output[y] -= 1

                grad_W2 = np.outer(hidden, grad_output)
                grad_hidden = np.dot(self.W2, grad_output)

                # Обновляем веса
                self.W2 -= learning_rate * grad_W2
                for word_idx in X:
                    self.W1[word_idx] -= learning_rate * grad_hidden / len(X)

            print(f"Epoch {epoch}, Loss: {loss / len(X_train)}")

# Обучаем три модели CBOW
embedding_dims = [100, 500, 1000]
cbow_models = {}

for dim in embedding_dims:
    print(f"\nTraining CBOW with d={dim}")
    model = Word2Vec(vocab_size, dim)
    model.train(X_cbow, y_cbow, epochs=10)
    cbow_models[dim] = model.W1  # Сохраняем эмбеддинги

# Подготовка данных (аналогично Skip-gram)
def prepare_sequences_cbow(text, word2idx, L=4):
    X, y = [], []
    for i in range(L, len(text)):
        context = text[i-L:i]
        target = text[i]
        X.append([word2idx[w] for w in context])
        y.append(word2idx[target])
    return torch.tensor(X), torch.tensor(y)

X_cbow_nn, y_cbow_nn = prepare_sequences_cbow(text, word2idx, L=4)

# Преобразуем в эмбеддинги
def embed_sequences_cbow(X, word2vec_matrix):
    embedded = []
    for seq in X:
        embedded_seq = word2vec_matrix[seq].flatten()  # Конкатенируем L векторов
        embedded.append(embedded_seq)
    return torch.tensor(np.array(embedded), dtype=torch.float32)

# Обучаем три модели
cbow_nn_models = {}
for dim in embedding_dims:
    print(f"\nTraining NN with CBOW d={dim}")
    X_embedded = embed_sequences_cbow(X_cbow_nn, cbow_models[dim])
    model = NextWordPredictor(dim * L, hidden_dim=500, vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_embedded)
        loss = criterion(outputs, y_cbow_nn)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    cbow_nn_models[dim] = model

test_sentence = "творчество достоевского одно"  # Должно быть L=4 слова (иначе дополняйте)
test_words = test_sentence.split()
test_indices = [word2idx[w] for w in test_words]

for dim in embedding_dims:
    print(f"\nCBOW Model with d={dim}:")
    model = cbow_nn_models[dim]
    word2vec_matrix = cbow_models[dim]
    
    # Получаем эмбеддинги
    embedded_test = word2vec_matrix[test_indices].flatten()
    embedded_test = torch.tensor(embedded_test, dtype=torch.float32).unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        output = model(embedded_test)
        predicted_idx = torch.argmax(output).item()
    
    print(f"Input: {test_sentence}")
    print(f"Predicted next word: {idx2word[predicted_idx]}")
