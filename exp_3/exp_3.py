# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 从外部文件加载名称列表
data_file = '数据.txt'
with open(data_file, 'r', encoding='utf-8') as f:
    code = f.read()
exec(code)  # 定义了 train_male_names, train_female_names, test_male_names, test_female_names

# 2. 构造训练集和测试集：男=0，女=1
train_data = [(name, 0) for name in train_male_names] + [(name, 1) for name in train_female_names]
test_data  = [(name, 0) for name in test_male_names]  + [(name, 1) for name in test_female_names]

# 3. 构建字符索引
all_chars = set("".join(name for name, _ in train_data + test_data))
char2idx = {char: idx+1 for idx, char in enumerate(all_chars)}  # 1 开始，0 用作 <PAD>
char2idx['<PAD>'] = 0
vocab_size = len(char2idx)
max_len = 3  # 假设名字不超过三字

def name_to_tensor(name):
    tensor = torch.zeros(max_len, dtype=torch.long)
    for i, ch in enumerate(name):
        tensor[i] = char2idx.get(ch, 0)
    return tensor

X_train = torch.stack([name_to_tensor(n) for n, _ in train_data])
y_train = torch.tensor([label for _, label in train_data])
X_test  = torch.stack([name_to_tensor(n) for n, _ in test_data])
y_test  = torch.tensor([label for _, label in test_data])

# 4. 定义 RNN 性别分类模型
class NameGenderClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        x, _  = self.lstm(self.embedding(x))
        out   = x[:, -1, :]              # 取最后一个时刻的隐藏状态
        out   = self.fc(out)
        return self.sigmoid(out)

# 5. 模型初始化与训练设置
embed_dim  = 10
hidden_dim = 20
model      = NameGenderClassifier(vocab_size, embed_dim, hidden_dim)
criterion  = nn.BCELoss()
optimizer  = optim.Adam(model.parameters(), lr=0.01)
epochs     = 50

# 6. 训练循环
for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss    = criterion(outputs, y_train.float())
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# 7. 测试集评估
model.eval()
with torch.no_grad():
    test_out = model(X_test).squeeze()
    preds    = (test_out > 0.5).int()
    acc      = (preds == y_test).float().mean()
    print(f"测试集准确率: {acc:.4f}")

# 8. 单独预测函数
def predict_name(name):
    model.eval()
    with torch.no_grad():
        t = name_to_tensor(name).unsqueeze(0)
        p = model(t).item()
    gender = "女" if p > 0.5 else "男"
    print(f"名字 {name} 预测性别: {gender} (概率: {p:.4f})")

# 9. 示例预测
predict_name("子涵")
predict_name("浩天")
predict_name("晓琳")
predict_name("振华")
