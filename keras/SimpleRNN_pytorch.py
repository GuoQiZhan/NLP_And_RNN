"""
SimpleRNN 模型在 IMDB 情感分析任务上的 PyTorch 实现

此文件是 SimpleRNN.py 的 PyTorch 版本，实现相同的功能。
Keras 和 PyTorch 的主要区别：
1. 数据加载：PyTorch 使用 DataLoader 和自定义 Dataset 类
2. 模型定义：PyTorch 需要继承 nn.Module 类并定义 forward 方法
3. 训练过程：PyTorch 需要手动编写训练循环

模型架构说明：
Embedding → SimpleRNN → Flatten → Dense(sigmoid)

SimpleRNN 的缺点：
    1. 梯度消失问题：在处理长序列时 SimpleRNN 容易遇到梯度消失问题
    2. 计算复杂度高：每个时间步的计算都依赖于前一个时间步的隐藏状态
    3. 只擅长处理短序列，会遗忘很久之前的信息

文件结构：
    1. 导入必要的 PyTorch 库
    2. 定义超参数
    3. 自定义 IMDB Dataset 类
    4. 构建 SimpleRNN 模型类
    5. 训练和评估函数
    6. 主程序执行流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# 导入 IMDB 数据集（使用 torchtext 或 keras 的数据集）
# 注意：这里使用 keras 的 IMDB 数据集，然后转换为 PyTorch 张量
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

# ==================== 定义超参数 ====================
'''
超参数说明（与 Keras 版本保持一致）：
vocabulary: unique words in the dictionary          词典里有10000个单词
embedding_dim: shape(x) = 32                        每个单词用32维向量表示
word_num: sequence length                           每个样本的序列长度为500
state_dim : shape(h) = 32                           隐藏层的维度为32
'''

VOCABULARY = 10000       # 词汇表大小，只保留数据集中最常见的10000个单词
EMBEDDING_DIM = 32       # 词嵌入维度，每个单词用一个32维的向量表示
WORD_NUM = 500           # 序列长度，每个评论被填充或截断为500个单词
STATE_DIM = 32           # SimpleRNN 隐藏状态的维度，即隐藏层有32个神经元
BATCH_SIZE = 32          # 批次大小
EPOCHS = 10              # 训练轮数
LEARNING_RATE = 0.001    # 学习率（对应 Keras 中的 RMSprop 学习率）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU

# ==================== 数据加载和预处理函数 ====================
def load_imdb_data():
    """
    加载和预处理 IMDB 数据集，与 Keras 版本保持一致

    返回:
        train_loader: 训练数据 DataLoader
        valid_loader: 验证数据 DataLoader
        test_loader: 测试数据 DataLoader
    """
    print("Loading IMDB dataset...\n")  # 打印加载数据集的提示信息
    # print("Loading IMDB dataset...")  # 打印加载数据集的提示信息

    # 加载 IMDB 数据集，只保留前10000个最常见的单词
    # x_train, x_test: 整数序列列表，每个整数代表一个单词（基于频率的索引）
    # y_train, y_test: 二进制标签列表（0表示负面评论，1表示正面评论）
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCABULARY)

    # 将序列填充到固定长度 WORD_NUM（500）
    # 长度不足500的序列会在前面补0，长度超过500的序列会被截断
    x_train = pad_sequences(x_train, maxlen=WORD_NUM)
    x_test = pad_sequences(x_test, maxlen=WORD_NUM)

    # 从测试集中划分出验证集（前5000个样本作为验证集，其余作为测试集）
    x_valid = x_test[:5000]   # 验证集特征
    y_valid = y_test[:5000]   # 验证集标签
    x_test = x_test[5000:]    # 测试集特征（剩余样本）
    y_test = y_test[5000:]    # 测试集标签（剩余样本）

    # 将 numpy 数组转换为 PyTorch 张量
    # 注意：需要转换为 long 类型（因为嵌入层需要整数索引）
    x_train_tensor = torch.LongTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 添加维度以匹配模型输出

    x_valid_tensor = torch.LongTensor(x_valid)
    y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1)

    x_test_tensor = torch.LongTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Data loaded and preprocessed \n")  # 打印数据加载完成提示信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader


# ==================== 定义 SimpleRNN 模型类 ====================
class SimpleRNNModel(nn.Module):
    """
    SimpleRNN 模型，与 Keras 版本功能相同

    模型架构：
    Embedding → SimpleRNN → Flatten → Dense(sigmoid)

    注意：PyTorch 的 RNN 默认返回 (output, hidden_state)
    设置 batch_first=True 使输入形状为 (batch, seq_len, feature)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_length):
        """
        初始化模型

        参数:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: RNN 隐藏状态维度
            seq_length: 序列长度
        """
        super(SimpleRNNModel, self).__init__()

        # 嵌入层：将整数索引转换为密集向量表示
        # 参数：vocab_size（输入维度），embedding_dim（输出维度）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # SimpleRNN 层：处理序列数据
        # 参数：input_size=embedding_dim, hidden_size=hidden_dim
        # batch_first=True: 输入形状为 (batch, seq_len, feature)
        # 注意：PyTorch 的 RNN 默认返回 (output, hidden_state)
        self.rnn = nn.RNN(input_size=embedding_dim,
                         hidden_size=hidden_dim,
                         batch_first=True,
                         nonlinearity='tanh')  # 默认使用 tanh 激活函数

        # 展平层：将 RNN 输出展平
        # RNN 输出形状: (batch, seq_len, hidden_dim)
        # 展平后形状: (batch, seq_len * hidden_dim)
        self.flatten = nn.Flatten()

        # 全连接输出层：单个神经元，sigmoid 激活函数
        # 输入维度: seq_len * hidden_dim
        # 输出维度: 1 (二分类概率)
        self.fc = nn.Linear(seq_length * hidden_dim, 1)

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

        # RNN 层: (batch, seq_len, embedding_dim) → (batch, seq_len, hidden_dim)
        # rnn_out 是每个时间步的隐藏状态
        rnn_out, _ = self.rnn(embedded)

        # 展平层: (batch, seq_len, hidden_dim) → (batch, seq_len * hidden_dim)
        flattened = self.flatten(rnn_out)

        # 全连接层 + sigmoid: (batch, seq_len * hidden_dim) → (batch, 1)
        # 使用 torch.sigmoid 而不是 nn.Sigmoid()，因为 BCEWithLogitsLoss 需要 logits
        output = torch.sigmoid(self.fc(flattened))
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


# ==================== 主程序 ====================
def main():
    """
    主函数：执行完整的训练和评估流程
    """
    # 1. 加载数据
    train_loader, valid_loader, test_loader = load_imdb_data()

    # 2. 创建模型
    model = SimpleRNNModel(
        vocab_size=VOCABULARY,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=STATE_DIM,
        seq_length=WORD_NUM
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
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # 6. 可选：保存模型
    # torch.save(trained_model.state_dict(), 'simplernn_model.pth')
    # print("Model saved to simplernn_model.pth")

if __name__ == "__main__":
    main()