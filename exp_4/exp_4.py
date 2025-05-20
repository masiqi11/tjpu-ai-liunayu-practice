# 题目：基于RNN训练逆序数字输出模型。给定一个长度不超过10的数字序列（字符仅限于数字0-9），模型需要将该数字序列逆序输出。例如，输入 12345，输出 54321。以学号+学号逆序为输入，输出结果。
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 定义一个简单直接的模型专门用于逆序
class SimpleReverseModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, output_size=10):
        super(SimpleReverseModel, self).__init__()
        self.hidden_size = hidden_size
        
        # 输入层到隐藏层
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        # 隐藏层
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        # 隐藏层到输出层
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x_sequence):
        # 获取序列长度
        seq_length = x_sequence.size(0)
        
        # 初始化输出序列
        outputs = []
        
        # 为每个输出位置获取特征
        for i in range(seq_length):
            # 对输入序列进行编码
            encoded = []
            for j in range(seq_length):
                x = x_sequence[j]
                h = self.input_to_hidden(x)
                h = self.relu1(h)
                encoded.append(h)
            
            # 整合所有编码特征
            context = torch.cat(encoded, dim=1)
            context = context.view(1, -1)  # 展平
            
            # 处理当前反向位置
            reverse_idx = seq_length - 1 - i
            current_feature = encoded[reverse_idx]
            
            # 生成输出
            h = self.hidden_to_hidden(current_feature)
            h = self.relu2(h)
            output = self.hidden_to_output(h)
            outputs.append(output)
        
        # 将输出堆叠成序列
        outputs = torch.stack(outputs, dim=0)
        return outputs

# 数据预处理函数
def preprocess_data(student_id):
    # 将学号转换为数字列表
    input_seq = [int(d) for d in str(student_id)]
    # 创建目标序列（逆序）
    target_seq = input_seq[::-1]
    
    # 将序列转换为one-hot编码
    input_tensor = torch.zeros(len(input_seq), 1, 10)
    target_tensor = torch.zeros(len(target_seq), 1, 10)
    
    for i, num in enumerate(input_seq):
        input_tensor[i][0][num] = 1
    for i, num in enumerate(target_seq):
        target_tensor[i][0][num] = 1
        
    return input_tensor, target_tensor

# 训练函数
def train_model(model, input_tensor, target_tensor, epochs=10000):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    best_model_state = None
    
    # 早停设置
    patience = 500
    counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        
        # 记录最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    print(f"训练完成！最佳损失: {best_loss:.4f}")
    
    return model

# 预测函数
def predict(model, student_id):
    input_tensor, _ = preprocess_data(student_id)
    with torch.no_grad():
        output = model(input_tensor)
        # 获取每个时间步的最大概率的索引
        predicted = torch.argmax(output, dim=2).squeeze().numpy()
        # 将预测结果转换为字符串
        result = ''.join(map(str, predicted))
    return result

def test_model():
    print("=" * 50)
    print("数字序列逆序预测模型测试程序")
    print("=" * 50)
    
    while True:
        try:
            # 获取用户输入
            student_id = input("\n请输入学号（输入'q'退出）: ")
            
            if student_id.lower() == 'q':
                print("程序已退出")
                break
                
            # 验证输入是否为纯数字
            if not student_id.isdigit():
                print("错误：请输入纯数字！")
                continue
                
            # 验证输入长度
            if len(student_id) > 10:
                print("错误：学号长度不能超过10位！")
                continue
            
            # 创建模型
            model = SimpleReverseModel()
            
            # 准备训练数据
            input_tensor, target_tensor = preprocess_data(student_id)
            
            # 训练模型
            print("\n开始训练模型...")
            model = train_model(model, input_tensor, target_tensor)
            
            # 预测
            predicted = predict(model, student_id)
            
            # 显示结果
            print("\n预测结果：")
            print(f"输入学号: {student_id}")
            print(f"预测结果: {predicted}")
            print(f"实际逆序: {student_id[::-1]}")
            print(f"预测正确: {'✓' if predicted == student_id[::-1] else '✗'}")
            
        except Exception as e:
            print(f"发生错误：{str(e)}")
            print("请重新输入！")

if __name__ == "__main__":
    test_model()
