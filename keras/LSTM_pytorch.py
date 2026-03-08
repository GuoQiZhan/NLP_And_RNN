"""
Long Short Term Memory (LSTM) 模型在 IMDB 情感分析任务上的 PyTorch 实现

此文件是 LSTM.py 的 PyTorch 版本，实现相同的功能。
Keras 和 PyTorch 的主要区别：
1. 数据加载：PyTorch 使用 DataLoader 和自定义 Dataset 类
2. 模型定义：PyTorch 需要继承 nn.Module 类并定义 forward 方法
3. 训练过程：PyTorch 需要手动编写训练循环

LSTM 相对于 SimpleRNN 的改进：
1. 解决梯度消失问题：通过门控机制控制信息流动
2. 能够处理长序列：记忆单元可以保持长期依赖
3. 四个门控结构：遗忘门、输入门、新值门、输出门

模型架构说明：
Embedding → LSTM (只取最后一个时间步) → Dense(sigmoid)

文件结构：
    1. 导入必要的 PyTorch 库
    2. 定义超参数
    3. 自定义 IMDB Dataset 类
    4. 构建 LSTM 模型类
    5. 训练和评估函数
    6. 主程序执行流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib
# 设置matplotlib后端为Agg，避免GUI问题（在无头环境中使用）
matplotlib.use('Agg')  # 在导入pyplot之前设置后端
import matplotlib.pyplot as plt

# 导入 IMDB 数据集（使用本地 aclImdb 数据集）
import numpy as np
import os
import re
from collections import Counter

# ==================== 定义超参数 ====================
'''
超参数说明（与 Keras 版本保持一致）：
vocabulary: unique words in the dictionary          词典里有10000个单词
embedding_dim: shape(x) = 32                        每个单词用32维向量表示
word_num: sequence length                           每个样本的序列长度为500
state_dim : shape(h) = 32                           LSTM隐藏状态的维度为32
注意：Keras LSTM 中使用了 dropout=0.2
'''

VOCABULARY = 10000       # 词汇表大小，只保留数据集中最常见的10000个单词
EMBEDDING_DIM = 32       # 词嵌入维度，每个单词用一个32维的向量表示
WORD_NUM = 500           # 序列长度，每个评论被填充或截断为500个单词
STATE_DIM = 32           # LSTM 隐藏状态的维度，即隐藏层有32个神经元
DROPOUT = 0.2           # Dropout 率，与 Keras 版本保持一致
BATCH_SIZE = 32          # 批次大小
EPOCHS = 10              # 训练轮数
LEARNING_RATE = 0.001    # 学习率（对应 Keras 中的 RMSprop 学习率）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU

# ==================== 辅助函数 ====================
def pad_sequences(sequences, maxlen, padding='pre', truncating='pre', value=0):
    """
    将整数序列列表填充到相同长度（自定义实现，替换Keras的pad_sequences）

    参数:
        sequences: 整数序列列表
        maxlen: 目标序列长度
        padding: 'pre'或'post'，表示在前面或后面填充
        truncating: 'pre'或'post'，表示从前面或后面截断
        value: 填充值

    返回:
        numpy数组，形状为 (len(sequences), maxlen)
    """
    result = np.full((len(sequences), maxlen), value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        if truncating == 'pre':
            trunc = seq[-maxlen:]
        else:
            trunc = seq[:maxlen]

        if padding == 'pre':
            result[i, -len(trunc):] = trunc
        else:
            result[i, :len(trunc)] = trunc
    return result


def text_to_sequence(text, word_index, max_words=VOCABULARY):
    """
    将文本转换为整数序列

    参数:
        text: 原始文本字符串
        word_index: 单词到索引的字典映射
        max_words: 词汇表大小限制

    返回:
        整数列表，单词索引序列
    """
    # 转换为小写并分词（简单分词）
    words = re.findall(r'\b\w+\b', text.lower())
    # 映射单词到索引，忽略未登录词（用0表示）
    # 注意：索引从1开始（0用于填充）
    sequence = []
    for word in words:
        if word in word_index:
            idx = word_index[word]
            if idx < max_words:  # 只保留前max_words个单词
                sequence.append(idx)
    return sequence


# ==================== 数据加载和预处理函数 ====================
def load_imdb_data():
    """
    加载和预处理本地 IMDB 数据集（从 aclImdb 目录）

    返回:
        train_loader: 训练数据 DataLoader
        valid_loader: 验证数据 DataLoader
        test_loader: 测试数据 DataLoader
    """
    print("Loading local IMDB dataset from aclImdb directory...\n")

    # 数据集路径
    base_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "aclImdb")
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")

    # 1. 加载词汇表
    vocab_path = os.path.join(base_path, "imdb.vocab")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]

    # 构建单词到索引的映射（索引从1开始，0用于填充）
    word_index = {word: i+1 for i, word in enumerate(vocab)}
    print(f"Loaded vocabulary with {len(vocab)} words")

    # 2. 加载训练数据
    print("Loading training data...")
    train_texts = []
    train_labels = []

    # 加载正面评论 (label=1)
    pos_dir = os.path.join(train_dir, "pos")
    for filename in os.listdir(pos_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            train_texts.append(text)
            train_labels.append(1)  # 正面评论标签为1

    # 加载负面评论 (label=0)
    neg_dir = os.path.join(train_dir, "neg")
    for filename in os.listdir(neg_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            train_texts.append(text)
            train_labels.append(0)  # 负面评论标签为0

    print(f"Loaded {len(train_texts)} training samples ({len(train_labels)} labels)")

    # 3. 加载测试数据
    print("Loading test data...")
    test_texts = []
    test_labels = []

    # 加载正面评论 (label=1)
    pos_dir = os.path.join(test_dir, "pos")
    for filename in os.listdir(pos_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            test_texts.append(text)
            test_labels.append(1)

    # 加载负面评论 (label=0)
    neg_dir = os.path.join(test_dir, "neg")
    for filename in os.listdir(neg_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            test_texts.append(text)
            test_labels.append(0)

    print(f"Loaded {len(test_texts)} test samples ({len(test_labels)} labels)")

    # 4. 将文本转换为整数序列
    print("Converting texts to sequences...")
    train_sequences = [text_to_sequence(text, word_index, VOCABULARY) for text in train_texts]
    test_sequences = [text_to_sequence(text, word_index, VOCABULARY) for text in test_texts]

    # 5. 序列填充
    print(f"Padding sequences to length {WORD_NUM}...")
    x_train = pad_sequences(train_sequences, maxlen=WORD_NUM, padding='pre', truncating='pre')
    x_test = pad_sequences(test_sequences, maxlen=WORD_NUM, padding='pre', truncating='pre')

    y_train = np.array(train_labels, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.float32)

    # 6. 从测试集中划分出验证集（前5000个样本作为验证集，与Keras版本保持一致）
    x_valid = x_test[:5000]   # 验证集特征
    y_valid = y_test[:5000]   # 验证集标签
    x_test = x_test[5000:]    # 测试集特征（剩余样本）
    y_test = y_test[5000:]    # 测试集标签（剩余样本）

    # 7. 将 numpy 数组转换为 PyTorch 张量
    x_train_tensor = torch.LongTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 添加维度以匹配模型输出

    x_valid_tensor = torch.LongTensor(x_valid)
    y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1)

    x_test_tensor = torch.LongTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    # 8. 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Data loaded and preprocessed \n")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader


# ==================== 定义 LSTM 模型类 ====================
class LSTMModel(nn.Module):
    """
    LSTM 模型，与 Keras 版本功能相同

    模型架构：
    Embedding → LSTM (只取最后一个时间步) → Dense(sigmoid)

    注意：PyTorch 的 LSTM 默认返回 (output, (hidden_state, cell_state))
    其中 output 包含所有时间步的隐藏状态
    设置 batch_first=True 使输入形状为 (batch, seq_len, feature)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate=0.0):
        """
        初始化模型

        参数:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM 隐藏状态维度
            dropout_rate: Dropout 率
        """
        super(LSTMModel, self).__init__()

        # 嵌入层：将整数索引转换为密集向量表示
        # 参数：vocab_size（输入维度），embedding_dim（输出维度）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM 层：处理序列数据
        # 参数：input_size=embedding_dim, hidden_size=hidden_dim
        # batch_first=True: 输入形状为 (batch, seq_len, feature)
        # dropout: 在 LSTM 层之间应用 dropout（如果 num_layers > 1）
        # 注意：PyTorch 的 LSTM 默认返回 (output, (hidden_state, cell_state))
        self.lstm = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_dim,
                           batch_first=True,
                           dropout=dropout_rate if dropout_rate > 0 else 0)

        # 全连接输出层：单个神经元，sigmoid 激活函数
        # 输入维度: hidden_dim（只取最后一个时间步的隐藏状态）
        # 输出维度: 1 (二分类概率)
        self.fc = nn.Linear(hidden_dim, 1)

        # Dropout 层（如果在 LSTM 之后需要额外的 dropout）
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, seq_length)

        返回:
            output: 预测概率，形状为 (batch_size, 1)
        """
        # 嵌入层: (batch, seq_len) → (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # LSTM 层: (batch, seq_len, embedding_dim) →
        # output: (batch, seq_len, hidden_dim) - 所有时间步的隐藏状态
        # (hidden_state, cell_state): 最后时间步的隐藏状态和细胞状态
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # 只取最后一个时间步的隐藏状态
        # hidden_state 的形状: (1, batch, hidden_dim)
        # 我们需要将其转换为 (batch, hidden_dim)
        last_hidden_state = hidden_state[-1]  # 取最后一个层（如果多层）

        # 可选的 dropout 层
        if self.dropout is not None:
            last_hidden_state = self.dropout(last_hidden_state)

        # 全连接层 + sigmoid: (batch, hidden_dim) → (batch, 1)
        # 使用 torch.sigmoid 而不是 nn.Sigmoid()，因为 BCEWithLogitsLoss 需要 logits
        # 但这里我们直接使用 sigmoid 输出概率，所以使用 BCELoss
        output = torch.sigmoid(self.fc(last_hidden_state))
        return output


# ==================== 训练函数 ====================
def train_model(model, train_loader, valid_loader, epochs, learning_rate):
    """
    训练模型

    参数:
        model: 要训练的模型
        train_loader: 训练数据 DataLoader
        valid_loader: 验证数据 DataLoader
        epochs: 训练轮数
        learning_rate: 学习率

    返回:
        model: 训练好的模型
        history: 训练历史记录
    """
    # 将模型移动到设备（GPU 或 CPU）
    model = model.to(DEVICE)

    # 定义损失函数和优化器
    # 使用二元交叉熵损失，与 Keras 的 binary_crossentropy 对应
    criterion = nn.BCELoss()

    # 使用 RMSprop 优化器，与 Keras 保持一致
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print("Training model... \n")  # 打印开始训练提示信息

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()  # 设置为训练模式
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            # 将数据移动到设备
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 统计训练损失和准确率
            train_loss += loss.item() * batch_x.size(0)

            # 计算准确率：预测概率 > 0.5 则为正类
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        # 计算平均训练损失和准确率
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        # ===== 验证阶段 =====
        model.eval()  # 设置为评估模式
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # 不需要计算梯度
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)

                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        # 计算平均验证损失和准确率
        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        # 打印每轮训练结果
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    return model, history


# ==================== 评估函数 ====================
def evaluate_model(model, test_loader):
    """
    评估模型在测试集上的性能

    参数:
        model: 要评估的模型
        test_loader: 测试数据 DataLoader

    返回:
        test_loss: 测试损失
        test_acc: 测试准确率
    """
    model.eval()  # 设置为评估模式
    criterion = nn.BCELoss()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    print("Evaluating model...")  # 打印开始评估提示信息

    with torch.no_grad():  # 不需要计算梯度
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            test_loss += loss.item() * batch_x.size(0)

            # 计算准确率：预测概率 > 0.5 则为正类
            predicted = (outputs > 0.5).float()
            test_correct += (predicted == batch_y).sum().item()
            test_total += batch_y.size(0)

    # 计算平均测试损失和准确率
    avg_test_loss = test_loss / test_total
    avg_test_acc = test_correct / test_total

    return avg_test_loss, avg_test_acc


# ==================== 可视化函数 ====================
def plot_training_history(history, test_loss=None, test_acc=None):
    """
    绘制训练曲线图像

    参数:
        history: 训练历史记录字典，包含 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        test_loss: 测试损失值（可选，保留用于向后兼容但在此图中未使用）
        test_acc: 测试准确率（可选），在图中显示为水平线
    """
    # 显式忽略 test_loss 参数（在此图中未使用，但保留用于向后兼容）
    _ = test_loss

    # 创建单一图形，使用双纵坐标轴
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 提取训练历史数据
    epochs = range(1, len(history['train_loss']) + 1)

    # 左侧纵坐标：Loss (训练损失)
    color = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Training Loss', color=color, fontsize=12)
    ax1.plot(epochs, history['train_loss'], color=color, linewidth=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 右侧纵坐标：Accuracy (训练准确率和测试准确率)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy', color=color, fontsize=12)

    # 训练准确率曲线
    ax2.plot(epochs, history['train_acc'], color=color, linewidth=2, label='Training Accuracy')

    # 测试准确率水平线（如果提供）
    if test_acc is not None:
        ax2.axhline(y=test_acc, color='tab:green', linestyle='--', linewidth=2,
                   label=f'Test Accuracy: {test_acc:.4f}')

    ax2.tick_params(axis='y', labelcolor=color)

    # 设置标题
    ax1.set_title('LSTM Training Curves', fontsize=14, fontweight='bold')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

    # 调整布局
    fig.tight_layout()

    # 保存图像
    plt.savefig('lstm_training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'lstm_training_curves.png'")

    # 显示图像（注释掉以避免在无GUI环境中出错）
    # plt.show()


# ==================== 主程序 ====================
def main():
    """
    主函数：执行完整的训练和评估流程
    """
    # 1. 加载数据
    train_loader, valid_loader, test_loader = load_imdb_data()

    # 2. 创建模型
    model = LSTMModel(
        vocab_size=VOCABULARY,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=STATE_DIM,
        dropout_rate=DROPOUT
    )

    # 3. 打印模型摘要
    print("Model summary:")
    print(model)
    print("\n")
    # 计算总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 4. 训练模型
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # 5. 评估模型
    test_loss, test_acc = evaluate_model(trained_model, test_loader)

    # 打印测试结果，与 Keras 版本格式保持一致
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # 6. 绘制训练曲线图像
    print("\nGenerating training curves...")
    plot_training_history(history, test_loss, test_acc)

    # 7. 可选：保存模型
    # torch.save(trained_model.state_dict(), 'lstm_model.pth')
    # print("Model saved to lstm_model.pth")

if __name__ == "__main__":
    main()