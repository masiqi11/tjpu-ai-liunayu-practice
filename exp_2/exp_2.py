import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import numpy as np
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

import os

# === 1. 从数据.txt 读取 train/test 数据 ===
def load_data_from_txt(file_path):
    # 使用 exec 安全地从文件中加载变量（假设数据结构规范）
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    data_scope = {}
    exec(code, data_scope)
    train_sentences = data_scope['train_sentences']
    train_labels = data_scope['train_labels']
    test_sentences = data_scope['test_sentences']
    test_labels = data_scope['test_labels']
    return train_sentences, train_labels, test_sentences, test_labels

train_sentences, train_labels, test_sentences, test_labels = load_data_from_txt("数据.txt")

# === 2. 分词处理 ===
tokenized_corpus = [list(jieba.cut(sentence)) for sentence in train_sentences]

# === 3. 训练 Word2Vec 模型 ===
word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# === 4. 句子向量平均表示函数 ===
def sentence_to_vector(sentence, model):
    words = list(jieba.cut(sentence))
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

X_train = np.array([sentence_to_vector(s, word2vec_model) for s in train_sentences])
X_test = np.array([sentence_to_vector(s, word2vec_model) for s in test_sentences])

# === 5. 标准化处理 ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. 转为张量 ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(np.array(train_labels, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(test_labels, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)

# === 7. 逻辑回归模型定义 ===
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# === 8. 模型训练 ===
model = LogisticRegressionModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === 9. 测试集评估 ===
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = (predictions > 0.5).float()
    accuracy = (predicted_labels == y_test_tensor).float().mean().item()
    print(f"测试集准确率: {accuracy:.4f}")