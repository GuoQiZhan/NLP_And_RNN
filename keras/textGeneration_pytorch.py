"""
Text Generation - PyTorch Implementation
1. Prepare Training Data
  使用莎士比亚诗歌数据集
  将所有字母小写
  每次读取长度为60的文本片段作为输入
  下一个字符作为标签
  将片段存储到segments列表中 标签存储到next_chars列表中
  步长stride = 3
2. Character to Vector
  创建一个包含文本中所有唯一字符的字典dictionary
  将每个字符映射到一个唯一的整数索引
  使用one-hot encoding将输入文本片段转换为二进制矩阵X
  将标签字符转换为整数索引并存储在数组y中
3. Build the Model
  使用单向LSTM层构建一个简单的神经网络模型
  输入层接受长度为60的文本片段
4. Train the Model
  使用categorical_crossentropy作为损失函数
  使用RMSprop优化器
  训练模型20个epoch
  lr = 0.01
5. Predict
  使用adjusting the multinomial distribution来生成文本

PyTorch implementation of the above plan.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置后端
import matplotlib.pyplot as plt

# ==================== Hyperparameters ====================
SEQ_LENGTH = 60        # 输入序列长度
STRIDE = 3             # 滑动窗口步长
BATCH_SIZE = 128       # 批次大小
EPOCHS = 20            # 训练轮数
LEARNING_RATE = 0.01   # 学习率
EMBEDDING_DIM = 50     # 字符嵌入维度（可调整）
HIDDEN_DIM = 128       # LSTM隐藏层维度
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Data Loading and Preprocessing ====================
def load_shakespeare_data(data_path):
    """
    加载莎士比亚数据集，全部转换为小写

    参数:
        data_path: 数据集文件路径

    返回:
        text: 全部小写的文本字符串
        chars: 唯一字符列表
        char_to_idx: 字符到索引的映射字典
        idx_to_char: 索引到字符的映射字典
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.lower()  # 全部转换为小写
    chars = sorted(list(set(text)))  # 唯一字符
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"Total characters in text: {len(text)}")
    print(f"Unique characters: {len(chars)}")
    print(f"First 100 characters: {text[:100]}")

    return text, chars, char_to_idx, idx_to_char

def create_sequences(text, char_to_idx, seq_length=60, stride=3):
    """
    创建输入序列和对应的标签

    参数:
        text: 输入文本
        char_to_idx: 字符到索引的映射
        seq_length: 序列长度
        stride: 滑动窗口步长

    返回:
        X: 输入序列数组，形状为 (num_samples, seq_length)
        y: 标签数组，形状为 (num_samples,)
    """
    segments = []
    next_chars = []

    for i in range(0, len(text) - seq_length, stride):
        segments.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])

    print(f"Created {len(segments)} sequences with stride {stride}")

    # 将字符转换为索引
    X = np.zeros((len(segments), seq_length), dtype=np.int64)
    y = np.zeros((len(segments),), dtype=np.int64)

    for i, segment in enumerate(segments):
        for t, char in enumerate(segment):
            X[i, t] = char_to_idx[char]
        y[i] = char_to_idx[next_chars[i]]

    return X, y

# ==================== Custom Dataset ====================
class ShakespeareDataset(Dataset):
    """莎士比亚数据集"""

    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== Model Definition ====================
class CharLSTM(nn.Module):
    """
    字符级LSTM模型用于文本生成

    模型架构:
    Embedding → LSTM → Dense(softmax)

    输入: (batch_size, seq_length) 整数索引
    输出: (batch_size, vocab_size) 下一个字符的概率分布
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_length):
        super(CharLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        # 嵌入层：将字符索引转换为密集向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM层：处理序列
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1  # 单向单层LSTM
        )

        # 全连接输出层：预测下一个字符的概率分布
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x形状: (batch_size, seq_length)

        # 嵌入层: (batch_size, seq_length) → (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)

        # LSTM层: (batch_size, seq_length, embedding_dim) →
        # output: (batch_size, seq_length, hidden_dim)
        # hidden: (hidden states)
        lstm_out, _ = self.lstm(embedded)

        # 只取最后一个时间步的输出: (batch_size, hidden_dim)
        last_output = lstm_out[:, -1, :]

        # 全连接层: (batch_size, hidden_dim) → (batch_size, vocab_size)
        # 注意：不使用softmax，因为CrossEntropyLoss内部会处理
        output = self.fc(last_output)

        return output

    def generate(self, seed, char_to_idx, idx_to_char, length=100, temperature=1.0):
        """
        生成文本

        参数:
            seed: 种子字符串
            char_to_idx: 字符到索引的映射
            idx_to_char: 索引到字符的映射
            length: 要生成的字符数
            temperature: 温度参数，控制随机性

        返回:
            生成的文本字符串
        """
        self.eval()
        generated = seed

        # 将种子转换为索引
        if len(seed) < self.seq_length:
            # 如果种子太短，填充或处理（这里简单处理）
            seed = seed + ' ' * (self.seq_length - len(seed))
        seed = seed[-self.seq_length:]  # 取最后seq_length个字符

        seed_indices = [char_to_idx[ch] for ch in seed]

        with torch.no_grad():
            for _ in range(length):
                # 准备输入
                x = torch.LongTensor(seed_indices).unsqueeze(0).to(DEVICE)

                # 前向传播
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                last_output = lstm_out[:, -1, :]
                logits = self.fc(last_output)

                # 应用温度参数
                logits = logits / temperature

                # 使用softmax获取概率分布
                probs = torch.softmax(logits, dim=-1)

                # 根据概率分布采样下一个字符
                next_char_idx = torch.multinomial(probs, 1).item()

                # 将生成的字符添加到结果中
                next_char = idx_to_char[next_char_idx]
                generated += next_char

                # 更新种子序列
                seed_indices = seed_indices[1:] + [next_char_idx]

        return generated

# ==================== Training Function ====================
def train_model(model, train_loader, val_loader, epochs, learning_rate, char_to_idx, idx_to_char):
    """
    训练模型

    参数:
        model: 要训练的模型
        train_loader: 训练数据DataLoader
        val_loader: 验证数据DataLoader
        epochs: 训练轮数
        learning_rate: 学习率
        char_to_idx: 字符到索引的映射
        idx_to_char: 索引到字符的映射

    返回:
        model: 训练好的模型
        history: 训练历史记录
    """
    model = model.to(DEVICE)

    # 损失函数：交叉熵损失（对应categorical_crossentropy）
    criterion = nn.CrossEntropyLoss()

    # 优化器：RMSprop（与Keras保持一致）
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print("Training model...\n")

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * batch_x.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        # 打印进度
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # 每个epoch后生成示例文本
        if (epoch + 1) % 5 == 0:
            print("\n--- Generated text after epoch {} ---".format(epoch + 1))
            seed = "the quick brown fox jumps over the lazy dog"
            generated = model.generate(
                seed[:model.seq_length],
                char_to_idx,
                idx_to_char,
                length=100,
                temperature=0.5
            )
            print(generated)
            print("---\n")

    return model, history

# ==================== Visualization Function ====================
def plot_training_history(history):
    """
    绘制训练曲线

    参数:
        history: 训练历史记录
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('text_generation_training_curves.png', dpi=300)
    print("Training curves saved to 'text_generation_training_curves.png'")

# ==================== Main Function ====================
def main():
    """主函数"""
    # 1. 加载数据
    data_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "shakespeare.txt")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please download Shakespeare dataset first.")
        return

    print("Loading Shakespeare dataset...")
    text, chars, char_to_idx, idx_to_char = load_shakespeare_data(data_path)

    # 2. 创建序列
    print("\nCreating sequences...")
    X, y = create_sequences(text, char_to_idx, SEQ_LENGTH, STRIDE)

    # 3. 划分训练集和验证集
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # 4. 创建DataLoader
    train_dataset = ShakespeareDataset(X_train, y_train)
    val_dataset = ShakespeareDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 创建模型
    vocab_size = len(chars)
    model = CharLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        seq_length=SEQ_LENGTH
    )

    print(f"\nModel created with vocabulary size: {vocab_size}")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 6. 训练模型
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char
    )

    # 7. 绘制训练曲线
    plot_training_history(history)

    # 8. 生成示例文本
    print("\n" + "="*50)
    print("Final text generation examples:")
    print("="*50)

    seeds = [
        "to be or not to be",
        "romeo, romeo, wherefore art thou",
        "shall i compare thee to a summer"
    ]

    for i, seed in enumerate(seeds):
        print(f"\n--- Example {i+1} (Seed: '{seed}') ---")
        # 确保种子长度合适
        if len(seed) < trained_model.seq_length:
            seed = seed + ' ' * (trained_model.seq_length - len(seed))
        seed = seed[:trained_model.seq_length]

        for temp in [0.5, 1.0, 1.5]:
            generated = trained_model.generate(
                seed,
                char_to_idx,
                idx_to_char,
                length=200,
                temperature=temp
            )
            print(f"\nTemperature {temp}:")
            print(generated[:200])  # 只打印前200个字符

    # 9. 保存模型
    model_path = 'char_lstm_model.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()