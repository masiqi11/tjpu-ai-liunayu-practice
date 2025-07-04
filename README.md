# 天津工业大学计算机学院
# 人工智能综合实践课程作业

本项目为天津工业大学计算机学院《人工智能综合实践》课程（刘那与老师）六次实践作业的完整代码与数据。每个实验均为独立子目录，便于查阅和运行。

---

## 目录结构

```
人工智能综合实践/
├── exp_1/      # 第一次实践：歌词分词与Word2Vec同义词分析
├── exp_2/      # 第二次实践：文本分类与逻辑回归
├── exp_3/      # 第三次实践：姓名性别预测（RNN/LSTM）
├── exp_4/      # 第四次实践：RNN逆序数字输出模型
├── exp_5/      # 第五次实践：Transformer逆序数字输出模型
├── exp_6/      # 第六次实践：Transformer Decoder中文姓名生成模型
├── 题目.txt    # 部分实验题目描述
```

---

## 各实验简介

### exp_1 歌词分词与Word2Vec同义词分析

- **内容**：对10k中文歌词进行分词、去停用词，训练Word2Vec模型，输出指定词的同义词。
- **主要文件**：
  - `exp_1.py`：主程序
  - `lyrics_10k.txt`：歌词语料
  - `stop_words.txt`：停用词表
  - `lyrics_word2vec.model`：训练好的词向量模型
- **依赖**：jieba、gensim

### exp_2 文本分类与逻辑回归

- **内容**：对文本数据进行分词、词向量平均，标准化后用PyTorch实现逻辑回归进行二分类。
- **主要文件**：
  - `exp_2.py`：主程序
  - `数据.txt`：训练/测试数据
- **依赖**：jieba、gensim、scikit-learn、torch

### exp_3 姓名性别预测（RNN/LSTM）

- **内容**：基于姓名字符的RNN/LSTM模型，预测姓名性别（男/女）。
- **主要文件**：
  - `exp_3.py`：主程序
  - `数据.txt`：姓名数据
- **依赖**：torch

### exp_4 RNN逆序数字输出模型

- **内容**：基于神经网络训练逆序数字输出模型。输入为长度不超过10的数字序列（如学号），输出为其逆序。
- **主要文件**：
  - `exp_4.py`：主程序
- **依赖**：torch、numpy
- **运行方式**：
  ```bash
  python exp_4.py
  ```
  按提示输入学号，程序会自动训练并输出逆序结果。

### exp_5 Transformer逆序数字输出模型

- **内容**：基于Transformer编解码架构训练逆序数字输出模型。输入为长度不超过10的数字序列（如学号），输出为其逆序。
- **主要文件**：
  - `exp-5.py`：主程序
  - `README.md`：实验详细说明
- **依赖**：torch、numpy
- **性能**：在测试集上达到99.82%的准确率
- **运行方式**：
  ```bash
  cd exp_5
  python exp-5.py
  ```
  程序会自动训练Transformer模型并接受用户输入进行测试。

### exp_6 Transformer Decoder中文姓名生成模型

- **内容**：基于Transformer decoder架构（2层，6个注意力头，隐层维度100）训练中文姓名生成模型。给定性别（男/女），生成合理的中文姓名。
- **主要文件**：
  - `exp-6.py`：主程序
  - `名字数据.txt`：训练数据
  - `README.md`：实验详细说明
- **依赖**：torch、numpy
- **模型参数**：2层Transformer decoder，6个注意力头，隐层维度102（≈100）
- **生成结果**：成功生成10个男性姓名和10个女性姓名
- **运行方式**：
  ```bash
  cd exp_6
  python exp-6.py
  ```
  程序会自动训练模型并生成姓名，同时提供交互式生成功能。

---

## 运行环境建议

- Python 3.8+
- 推荐使用虚拟环境（如conda）
- 依赖包见各实验代码头部注释

---

## 致谢

本项目为天津工业大学计算机学院刘那与老师《人工智能综合实践》课程作业，感谢老师的指导与同学们的交流。

---

如需详细实验说明，请查阅各子目录下的 `题目.pdf` 或 `题目.txt` 文件。 