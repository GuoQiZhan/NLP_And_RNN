"""
Seq2Seq
  1.Datasets
    Preprocessing: to lower case, remove punctuation, tokenize, and convert to sequences of integers.
  2. Tokenization and Build Dictionary
    input_texts => [English_Tokenizer] => input_tokens
    target_texts => [Chinese_Tokenizer] => target_tokens

    use 2 different tokenizers for the 2 languages
    Then build 2 different dictionaries

    use Tokenization in the char-level, not word-level, because the number of unique characters is much smaller than the number of unique words, which can reduce the complexity of the model and improve the performance.

  3. One-hot Encoding
  4. Training the Seq2Seq Model
    - Encoder: LSTM
    - Decoder: LSTM 

    使用'\t'作为输入文本和目标文本之间的起始符 使用'\n'作为目标文本的结束标志。
    Loss: CorssEntropy Loss

  5. Seq2Seq
    encoder_inputs: InputLayer(English sentences) -> encoder_lstm: LSTM -> decoder_lstm: LSTM 
    decoder_inputs: InputLayer(Part of Chinese sentences) -> decoder_lstm: LSTM
    decoder_lstm -> dense: Dense

"""
"""
Seq2Seq (Sequence-to-Sequence) 模型用于英中翻译任务的 PyTorch 实现

此文件是 Sequence_to_Sequence.py 的 PyTorch 版本，实现相同的功能。
Keras 和 PyTorch 的主要区别：
1. 数据加载：PyTorch 使用 DataLoader 和自定义 Dataset 类
2. 模型定义：PyTorch 需要继承 nn.Module 类并定义 forward 方法
3. 训练过程：PyTorch 需要手动编写训练循环

Seq2Seq 模型说明：
1. 编码器：LSTM，将输入英文句子编码为上下文向量
2. 解码器：LSTM，使用上下文向量和起始符生成中文翻译
3. 字符级分词：对英文和中文分别使用字符级分词，减少词汇表大小
4. 特殊标记：使用 '\t' 作为起始符，'\n' 作为结束符

模型架构说明：
Encoder: InputLayer(English sentences) -> encoder_lstm: LSTM -> context vector
Decoder: InputLayer(Chinese sentences with start token) -> decoder_lstm: LSTM -> Dense(softmax)

文件结构：
    1. 导入必要的 PyTorch 库
    2. 定义超参数
    3. 自定义 Translation Dataset 类
    4. 构建 Seq2Seq 模型类（Encoder, Decoder, Seq2Seq）
    5. 训练和评估函数
    6. 主程序执行流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
import matplotlib
# 设置matplotlib后端为Agg，避免GUI问题（在无头环境中使用）
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import random

# ==================== 定义超参数 ====================
"""
超参数说明：
MAX_INPUT_LENGTH: 英文句子最大长度（字符数）
MAX_TARGET_LENGTH: 中文句子最大长度（字符数）
BATCH_SIZE: 批次大小
EMBEDDING_DIM: 词嵌入维度（字符嵌入维度）
HIDDEN_DIM: LSTM 隐藏状态维度
DROPOUT: Dropout 率
EPOCHS: 训练轮数
LEARNING_RATE: 学习率
TEACHER_FORCING_RATIO: 教师强制比例（训练时使用真实目标作为下一输入的概率）
"""

MAX_INPUT_LENGTH = 50      # 英文句子最大长度（字符数）
MAX_TARGET_LENGTH = 50     # 中文句子最大长度（字符数）
BATCH_SIZE = 32            # 批次大小
EMBEDDING_DIM = 128        # 字符嵌入维度
HIDDEN_DIM = 256           # LSTM 隐藏状态维度
DROPOUT = 0.2              # Dropout 率
EPOCHS = 20                # 训练轮数
LEARNING_RATE = 0.001      # 学习率
TEACHER_FORCING_RATIO = 0.5  # 教师强制比例

# 特殊标记
PAD_TOKEN = '<PAD>'        # 填充标记
START_TOKEN = '\t'         # 起始标记（与注释一致）
END_TOKEN = '\n'           # 结束标记（与注释一致）
UNK_TOKEN = '<UNK>'        # 未知字符标记

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU

# ==================== 辅助函数 ====================
def preprocess_text(text):
    """
    文本预处理：转换为小写，移除多余空格
    参数:
        text: 原始文本字符串
    返回:
        预处理后的文本
    """
    text = text.lower().strip()
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    return text

def pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
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

# ==================== 数据加载和预处理函数 ====================
def load_translation_data(data_path, max_samples=50000):
    """
    加载和预处理英中翻译数据集

    参数:
        data_path: 数据集文件路径
        max_samples: 最大样本数（限制数据集大小）

    返回:
        input_texts: 英文句子列表
        target_texts: 中文句子列表
    """
    print(f"Loading translation data from {data_path}...")

    input_texts = []  # 英文句子
    target_texts = [] # 中文句子

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 限制样本数量
    if max_samples:
        lines = lines[:max_samples]

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            english = parts[0].strip()
            chinese = parts[1].strip()
            # 简单的数据清洗：移除空句子和过长句子
            if english and chinese and len(english) < 100 and len(chinese) < 100:
                input_texts.append(preprocess_text(english))
                target_texts.append(chinese)  # 中文保持原样（不需要小写）

    print(f"Loaded {len(input_texts)} translation pairs")
    return input_texts, target_texts

def build_char_vocab(texts, max_chars=None):
    """
    构建字符词汇表

    参数:
        texts: 文本列表
        max_chars: 最大字符数量限制

    返回:
        char2idx: 字符到索引的字典
        idx2char: 索引到字符的列表
    """
    # 统计所有字符频率
    char_counter = Counter()
    for text in texts:
        char_counter.update(text)

    # 按频率排序
    sorted_chars = [char for char, count in char_counter.most_common()]

    # 限制词汇表大小（如果指定）
    if max_chars:
        sorted_chars = sorted_chars[:max_chars-4]  # 保留位置给特殊标记

    # 构建词汇表（从1开始索引，0用于填充）
    char2idx = {PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, UNK_TOKEN: 3}
    idx2char = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

    # 添加普通字符
    for char in sorted_chars:
        if char not in char2idx:
            char2idx[char] = len(idx2char)
            idx2char.append(char)

    print(f"Built vocabulary with {len(char2idx)} characters")
    return char2idx, idx2char

def texts_to_sequences(texts, char2idx, max_length, add_start_end=False):
    """
    将文本列表转换为整数序列

    参数:
        texts: 文本列表
        char2idx: 字符到索引的字典
        max_length: 最大序列长度
        add_start_end: 是否添加起始和结束标记

    返回:
        sequences: 整数序列列表
    """
    sequences = []
    for text in texts:
        seq = []
        if add_start_end:
            seq.append(char2idx[START_TOKEN])

        for char in text:
            seq.append(char2idx.get(char, char2idx[UNK_TOKEN]))

        if add_start_end:
            seq.append(char2idx[END_TOKEN])

        # 截断到最大长度（考虑起始和结束标记）
        if max_length:
            if add_start_end:
                max_seq_len = max_length + 2  # 包括起始和结束标记
            else:
                max_seq_len = max_length
            seq = seq[:max_seq_len]

        sequences.append(seq)

    return sequences

# ==================== 自定义 Dataset 类 ====================
class TranslationDataset(Dataset):
    """
    英中翻译数据集类
    """

    def __init__(self, input_texts, target_texts,
                 input_char2idx, target_char2idx,
                 max_input_length, max_target_length):
        """
        初始化数据集

        参数:
            input_texts: 输入文本（英文）列表
            target_texts: 目标文本（中文）列表
            input_char2idx: 输入字符到索引的字典
            target_char2idx: 目标字符到索引的字典
            max_input_length: 输入序列最大长度
            max_target_length: 目标序列最大长度
        """
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.input_char2idx = input_char2idx
        self.target_char2idx = target_char2idx
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # 将文本转换为序列
        self.input_sequences = texts_to_sequences(
            input_texts, input_char2idx, max_input_length, add_start_end=False)
        self.target_sequences = texts_to_sequences(
            target_texts, target_char2idx, max_target_length, add_start_end=True)

        # 填充序列
        self.padded_inputs = pad_sequences(
            self.input_sequences, maxlen=max_input_length, padding='post', truncating='post')
        self.padded_targets = pad_sequences(
            self.target_sequences, maxlen=max_target_length+2, padding='post', truncating='post')  # +2 因为起始和结束标记

        # 转换为张量
        self.input_tensor = torch.LongTensor(self.padded_inputs)
        self.target_tensor = torch.LongTensor(self.padded_targets)

        print(f"Dataset created with {len(self)} samples")
        print(f"Input shape: {self.input_tensor.shape}")
        print(f"Target shape: {self.target_tensor.shape}")

    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, idx):
        return self.input_tensor[idx], self.target_tensor[idx]

# ==================== 模型定义 ====================
class Encoder(nn.Module):
    """
    编码器：将输入序列编码为上下文向量
    """

    def __init__(self, input_size, embedding_dim, hidden_dim, dropout_rate=0.0):
        """
        初始化编码器

        参数:
            input_size: 输入词汇表大小（英文字符数量）
            embedding_dim: 嵌入维度
            hidden_dim: LSTM 隐藏状态维度
            dropout_rate: Dropout 率
        """
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0,
            bidirectional=False  # 单层单向LSTM
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 用于将LSTM隐藏状态转换为上下文向量（如果需要）
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, seq_length)

        返回:
            outputs: LSTM 所有时间步的输出，形状为 (batch_size, seq_length, hidden_dim)
            hidden_state: 最后时间步的隐藏状态，形状为 (1, batch_size, hidden_dim)
            cell_state: 最后时间步的细胞状态，形状为 (1, batch_size, hidden_dim)
        """
        # 嵌入层: (batch, seq_len) → (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)

        if self.dropout is not None:
            embedded = self.dropout(embedded)

        # LSTM 层: (batch, seq_len, embedding_dim) →
        # outputs: (batch, seq_len, hidden_dim)
        # hidden_state, cell_state: (1, batch, hidden_dim)
        outputs, (hidden_state, cell_state) = self.lstm(embedded)

        return outputs, hidden_state, cell_state

class Decoder(nn.Module):
    """
    解码器：使用上下文向量生成目标序列
    """

    def __init__(self, output_size, embedding_dim, hidden_dim, dropout_rate=0.0):
        """
        初始化解码器

        参数:
            output_size: 输出词汇表大小（中文字符数量）
            embedding_dim: 嵌入维度
            hidden_dim: LSTM 隐藏状态维度（与编码器相同）
            dropout_rate: Dropout 率
        """
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.output_size = output_size
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden_state, cell_state):
        """
        前向传播（单步解码）

        参数:
            x: 当前输入字符张量，形状为 (batch_size, 1)
            hidden_state: LSTM 隐藏状态，形状为 (1, batch_size, hidden_dim)
            cell_state: LSTM 细胞状态，形状为 (1, batch_size, hidden_dim)

        返回:
            output: 下一个字符的预测概率分布，形状为 (batch_size, output_size)
            hidden_state: 更新后的隐藏状态
            cell_state: 更新后的细胞状态
        """
        # 嵌入层: (batch, 1) → (batch, 1, embedding_dim)
        embedded = self.embedding(x)

        if self.dropout is not None:
            embedded = self.dropout(embedded)

        # LSTM 层: (batch, 1, embedding_dim) → (batch, 1, hidden_dim)
        # hidden_state, cell_state: (1, batch, hidden_dim)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded, (hidden_state, cell_state))

        # 全连接层: (batch, 1, hidden_dim) → (batch, 1, output_size)
        # 然后去掉时间步维度: (batch, output_size)
        output = self.fc(lstm_out.squeeze(1))

        return output, hidden_state, cell_state

class Seq2Seq(nn.Module):
    """
    Seq2Seq 模型：组合编码器和解码器
    """

    def __init__(self, encoder, decoder, device):
        """
        初始化 Seq2Seq 模型

        参数:
            encoder: 编码器实例
            decoder: 解码器实例
            device: 设备（CPU/GPU）
        """
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        """
        前向传播（训练模式）

        参数:
            source: 源序列（英文），形状为 (batch_size, source_len)
            target: 目标序列（中文），形状为 (batch_size, target_len)
            teacher_forcing_ratio: 教师强制比例

        返回:
            outputs: 所有时间步的输出，形状为 (batch_size, target_len-1, output_size)
        """
        batch_size = source.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.output_size

        # 存储解码器输出
        outputs = torch.zeros(batch_size, target_len-1, target_vocab_size).to(self.device)

        # 编码器前向传播
        _, hidden_state, cell_state = self.encoder(source)

        # 解码器的第一个输入是起始标记
        decoder_input = target[:, 0].unsqueeze(1)  # (batch, 1)

        # 逐步解码
        for t in range(1, target_len):
            # 解码器单步前向传播
            output, hidden_state, cell_state = self.decoder(
                decoder_input, hidden_state, cell_state)

            # 存储输出（跳过起始标记）
            outputs[:, t-1, :] = output

            # 决定下一个输入：教师强制或模型预测
            teacher_force = random.random() < teacher_forcing_ratio

            # 获取预测的下一个字符（概率最大的索引）
            top1 = output.argmax(1).unsqueeze(1)  # (batch, 1)

            # 如果是教师强制，使用真实目标；否则使用模型预测
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def predict(self, source, max_length=50, start_token_idx=1):
        """
        预测（推理模式）

        参数:
            source: 源序列（英文），形状为 (batch_size, source_len)
            max_length: 最大生成长度
            start_token_idx: 起始标记的索引

        返回:
            outputs: 预测的序列，形状为 (batch_size, max_length)
        """
        batch_size = source.size(0)

        # 编码器前向传播
        _, hidden_state, cell_state = self.encoder(source)

        # 解码器的第一个输入是起始标记
        decoder_input = torch.tensor([[start_token_idx]] * batch_size).to(self.device)  # (batch, 1)

        # 存储生成的序列
        outputs = torch.zeros(batch_size, max_length).long().to(self.device)

        # 逐步解码
        for t in range(max_length):
            # 解码器单步前向传播
            output, hidden_state, cell_state = self.decoder(
                decoder_input, hidden_state, cell_state)

            # 获取预测的下一个字符（概率最大的索引）
            top1 = output.argmax(1).unsqueeze(1)  # (batch, 1)

            # 存储预测结果
            outputs[:, t] = top1.squeeze(1)

            # 更新解码器输入
            decoder_input = top1

        return outputs

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, teacher_forcing_ratio):
    """
    训练一个epoch

    参数:
        model: 要训练的模型
        dataloader: 训练数据 DataLoader
        criterion: 损失函数
        optimizer: 优化器
        teacher_forcing_ratio: 教师强制比例

    返回:
        epoch_loss: 平均损失
        epoch_acc: 平均准确率（字符级别）
    """
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    for batch_idx, (source, target) in enumerate(dataloader):
        source = source.to(DEVICE)
        target = target.to(DEVICE)

        # 前向传播
        output = model(source, target, teacher_forcing_ratio)

        # 准备目标：跳过起始标记，只保留需要预测的部分
        # target: (batch, target_len) -> target[:, 1:] (batch, target_len-1)
        target_output = target[:, 1:]

        # 计算损失
        # output: (batch, target_len-1, vocab_size) -> (batch * (target_len-1), vocab_size)
        # target_output: (batch, target_len-1) -> (batch * (target_len-1))
        loss = criterion(
            output.reshape(-1, output.shape[2]),
            target_output.reshape(-1)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        # 统计损失
        epoch_loss += loss.item()

        # 计算准确率（字符级别）
        predicted = output.argmax(2)  # (batch, target_len-1)
        correct = (predicted == target_output).sum().item()
        total = target_output.numel()

        epoch_correct += correct
        epoch_total += total

        # 打印批次进度（每10个批次）
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return epoch_loss / len(dataloader), epoch_correct / epoch_total

def evaluate(model, dataloader, criterion):
    """
    评估模型

    参数:
        model: 要评估的模型
        dataloader: 评估数据 DataLoader
        criterion: 损失函数

    返回:
        epoch_loss: 平均损失
        epoch_acc: 平均准确率（字符级别）
    """
    model.eval()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    with torch.no_grad():
        for source, target in dataloader:
            source = source.to(DEVICE)
            target = target.to(DEVICE)

            # 前向传播（不使用教师强制）
            output = model(source, target, teacher_forcing_ratio=0)

            # 准备目标
            target_output = target[:, 1:]

            # 计算损失
            loss = criterion(
                output.reshape(-1, output.shape[2]),
                target_output.reshape(-1)
            )

            # 统计损失
            epoch_loss += loss.item()

            # 计算准确率
            predicted = output.argmax(2)
            correct = (predicted == target_output).sum().item()
            total = target_output.numel()

            epoch_correct += correct
            epoch_total += total

    return epoch_loss / len(dataloader), epoch_correct / epoch_total

# ==================== 主程序 ====================
def main():
    """
    主函数：执行完整的训练和评估流程
    """
    print("=" * 60)
    print("Seq2Seq 英中翻译模型 - PyTorch 实现")
    print("=" * 60)

    # 1. 加载数据
    data_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "cmn-eng", "cmn.txt")
    input_texts, target_texts = load_translation_data(data_path, max_samples=50000)

    # 2. 构建词汇表
    print("\n构建英文字符词汇表...")
    input_char2idx, input_idx2char = build_char_vocab(input_texts, max_chars=200)

    print("\n构建中文字符词汇表...")
    target_char2idx, target_idx2char = build_char_vocab(target_texts, max_chars=2000)

    # 3. 创建数据集和数据加载器
    print("\n创建数据集...")
    dataset = TranslationDataset(
        input_texts, target_texts,
        input_char2idx, target_char2idx,
        MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )

    # 划分训练集和验证集（80%训练，20%验证）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 4. 创建模型
    print("\n创建模型...")
    encoder = Encoder(
        input_size=len(input_char2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT
    )

    decoder = Decoder(
        output_size=len(target_char2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT
    )

    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # 5. 打印模型摘要
    print("\n模型摘要:")
    print(model)
    print("\n")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 6. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. 训练模型
    print("\n开始训练...")
    print("=" * 60)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, TEACHER_FORCING_RATIO)

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印结果
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

        # 保存最佳模型
        if epoch == 0 or val_loss < min(history['val_loss'][:-1]):
            torch.save(model.state_dict(), 'seq2seq_model.pth')
            print("模型已保存为 seq2seq_model.pth")

    print("\n训练完成!")
    print("=" * 60)

    # 8. 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_history(history)

    # 9. 示例翻译
    print("\n示例翻译:")
    print("-" * 40)

    # 切换到评估模式
    model.eval()

    # 选择几个示例
    sample_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))

    for idx in sample_indices:
        source, target = val_dataset[idx]
        source = source.unsqueeze(0).to(DEVICE)
        target = target.unsqueeze(0).to(DEVICE)

        # 获取预测
        prediction = model.predict(source, max_length=MAX_TARGET_LENGTH+2)

        # 将索引转换为字符
        source_text = ''.join([input_idx2char[i] for i in source[0].cpu().numpy() if i != 0])
        target_text = ''.join([target_idx2char[i] for i in target[0].cpu().numpy() if i not in [0, 1, 2]])  # 忽略特殊标记
        predicted_text = ''.join([target_idx2char[i] for i in prediction[0].cpu().numpy() if i not in [0, 1, 2]])

        # 打印结果
        print(f"英文: {source_text}")
        print(f"中文 (真实): {target_text}")
        print(f"中文 (预测): {predicted_text}")
        print("-" * 40)

def plot_training_history(history):
    """
    绘制训练曲线图像

    参数:
        history: 训练历史记录字典，包含 'train_loss', 'train_acc', 'val_loss', 'val_acc'
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
    plt.savefig('seq2seq_training_curves.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存为 seq2seq_training_curves.png")

if __name__ == "__main__":
    main()