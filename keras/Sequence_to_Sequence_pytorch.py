"""
Seq2Seq
  1.Datasets
    Preprocessing: to lower case, remove punctuation, tokenize, and convert to sequences of integers.
  2. Tokenization and Build Dictionary
    input_texts => [English_Tokenizer] => input_tokens
    target_texts => [Chinese_Tokenizer] => target_tokens

    use 2 different tokenizers for the 2 languages
    Then build 2 different dictionaries

    use Tokenization in the word-level for English and char-level for Chinese. English uses word-level tokenization (space and punctuation), Chinese uses character-level to keep vocabulary size manageable.

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
3. 分词策略：英文使用词级分词（按空格和标点），中文使用字符级分词，平衡词汇表大小和语义信息
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
MAX_INPUT_LENGTH: 英文句子最大长度（词数）
MAX_TARGET_LENGTH: 中文句子最大长度（字符数）
BATCH_SIZE: 批次大小
EMBEDDING_DIM: 词嵌入维度
HIDDEN_DIM: LSTM 隐藏状态维度
DROPOUT: Dropout 率
EPOCHS: 训练轮数
LEARNING_RATE: 学习率
TEACHER_FORCING_RATIO: 教师强制比例（训练时使用真实目标作为下一输入的概率）
"""

MAX_INPUT_LENGTH = 20      # 英文句子最大长度（词数），原50字符数，现在减少
MAX_TARGET_LENGTH = 50     # 中文句子最大长度（字符数）
BATCH_SIZE = 32            # 批次大小
EMBEDDING_DIM = 256        # 词嵌入维度（从128增加到256，词级需要更大维度）
HIDDEN_DIM = 512           # LSTM 隐藏状态维度（从256增加到512）
DROPOUT = 0.3              # Dropout 率（从0.2增加到0.3，词级需要更多正则化）
EPOCHS = 35                # 训练轮数（从25增加到35，更深的模型需要更多训练）
LEARNING_RATE = 0.0003     # 学习率（从0.0005减小到0.0003，更深的模型需要更小心训练）
TEACHER_FORCING_RATIO = 0.5  # 教师强制比例

# 特殊标记
PAD_TOKEN = '<PAD>'        # 填充标记
START_TOKEN = '\t'         # 起始标记（与注释一致）
END_TOKEN = '\n'           # 结束标记（与注释一致）
UNK_TOKEN = '<UNK>'        # 未知字符标记

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU

# ==================== 模型权重初始化函数 ====================
def init_weights(module):
    """
    初始化模型权重
    对LSTM和线性层使用Xavier均匀初始化
    对嵌入层使用正态分布初始化
    """
    if isinstance(module, nn.Embedding):
        # 嵌入层初始化
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
    elif isinstance(module, nn.Linear):
        # 线性层初始化
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.LSTM):
        # LSTM权重初始化
        for name, param in module.named_parameters():
            if 'weight' in name:
                # 对LSTM权重使用正交初始化
                for i in range(0, param.size(0), module.hidden_size):
                    nn.init.orthogonal_(param.data[i:i+module.hidden_size])
            elif 'bias' in name:
                # LSTM偏置：遗忘门偏置初始化为1（帮助梯度流动）
                nn.init.zeros_(param.data)
                # 设置遗忘门偏置为1（如果存在足够的元素）
                if param.size(0) >= module.hidden_size * 4:
                    param.data[module.hidden_size:module.hidden_size*2].fill_(1)

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

def tokenize_english_word_level(text):
    """
    英文词级分词：按空格和常见标点分词
    参数:
        text: 英文文本字符串（已小写化）
    返回:
        词列表
    """
    # 使用正则表达式按空格和标点分词，保留词素
    # 匹配字母数字字符序列和单独标点
    tokens = re.findall(r"\w+|[^\w\s]", text)
    # 过滤空字符串
    tokens = [token for token in tokens if token.strip()]
    return tokens

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

def build_vocab(texts, max_items=None, tokenizer=None):
    """
    构建词汇表（字符级或词级）

    参数:
        texts: 文本列表
        max_items: 最大词汇表大小限制
        tokenizer: 分词器函数，如果不提供则按字符分割

    返回:
        token2idx: 标记到索引的字典
        idx2token: 索引到标记的列表
    """
    # 统计所有标记频率
    token_counter = Counter()
    for text in texts:
        if tokenizer:
            tokens = tokenizer(text)
        else:
            tokens = text  # 字符级：字符串可迭代
        token_counter.update(tokens)

    # 按频率排序
    sorted_tokens = [token for token, count in token_counter.most_common()]

    # 限制词汇表大小（如果指定）
    if max_items:
        sorted_tokens = sorted_tokens[:max_items-4]  # 保留位置给特殊标记

    # 构建词汇表（从1开始索引，0用于填充）
    token2idx = {PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, UNK_TOKEN: 3}
    idx2token = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

    # 添加普通标记
    for token in sorted_tokens:
        if token not in token2idx:
            token2idx[token] = len(idx2token)
            idx2token.append(token)

    vocab_type = "word" if tokenizer else "character"
    print(f"Built {vocab_type} vocabulary with {len(token2idx)} items")
    return token2idx, idx2token

# 保持向后兼容的别名（接受 max_chars 参数）
def build_char_vocab(texts, max_chars=None, tokenizer=None):
    """
    向后兼容的字符词汇表构建函数
    接受 max_chars 参数（转换为 max_items）
    """
    # 如果提供了 tokenizer，使用它；否则默认为字符级分词
    actual_tokenizer = tokenizer
    return build_vocab(texts, max_items=max_chars, tokenizer=actual_tokenizer)

def texts_to_sequences(texts, token2idx, max_length, add_start_end=False, tokenizer=None):
    """
    将文本列表转换为整数序列

    参数:
        texts: 文本列表
        token2idx: 标记到索引的字典
        max_length: 最大序列长度
        add_start_end: 是否添加起始和结束标记
        tokenizer: 分词器函数，如果不提供则按字符分割

    返回:
        sequences: 整数序列列表
    """
    sequences = []
    for text in texts:
        seq = []
        if add_start_end:
            seq.append(token2idx[START_TOKEN])

        # 使用分词器或按字符分割
        if tokenizer:
            tokens = tokenizer(text)
        else:
            tokens = text  # 字符级：字符串可迭代

        for token in tokens:
            seq.append(token2idx.get(token, token2idx[UNK_TOKEN]))

        if add_start_end:
            seq.append(token2idx[END_TOKEN])

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
                 input_token2idx, target_token2idx,
                 max_input_length, max_target_length,
                 input_tokenizer=None, target_tokenizer=None):
        """
        初始化数据集

        参数:
            input_texts: 输入文本（英文）列表
            target_texts: 目标文本（中文）列表
            input_token2idx: 输入标记到索引的字典
            target_token2idx: 目标标记到索引的字典
            max_input_length: 输入序列最大长度
            max_target_length: 目标序列最大长度
            input_tokenizer: 输入文本分词器函数（如不提供则按字符分割）
            target_tokenizer: 目标文本分词器函数（如不提供则按字符分割）
        """
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.input_token2idx = input_token2idx
        self.target_token2idx = target_token2idx
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer

        # 将文本转换为序列
        self.input_sequences = texts_to_sequences(
            input_texts, input_token2idx, max_input_length,
            add_start_end=False, tokenizer=input_tokenizer)
        self.target_sequences = texts_to_sequences(
            target_texts, target_token2idx, max_target_length,
            add_start_end=True, tokenizer=target_tokenizer)

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
class Attention(nn.Module):
    """
    Bahdanau注意力机制（加性注意力）
    用于Seq2Seq模型，提高解码器对编码器输出的关注
    """

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        """
        初始化注意力机制

        参数:
            encoder_hidden_dim: 编码器隐藏状态维度
            decoder_hidden_dim: 解码器隐藏状态维度
        """
        super(Attention, self).__init__()
        self.W_a = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        self.U_a = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.v_a = nn.Linear(decoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        前向传播

        参数:
            decoder_hidden: 解码器隐藏状态，形状为 (batch_size, decoder_hidden_dim)
            encoder_outputs: 编码器所有时间步的输出，形状为 (batch_size, seq_len, encoder_hidden_dim)

        返回:
            context_vector: 上下文向量，形状为 (batch_size, encoder_hidden_dim)
            attention_weights: 注意力权重，形状为 (batch_size, seq_len)
        """
        # decoder_hidden: (batch, decoder_hidden_dim) -> (batch, 1, decoder_hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1)

        # 计算注意力分数
        # encoder_outputs: (batch, seq_len, encoder_hidden_dim)
        # score = v_a * tanh(W_a * encoder_outputs + U_a * decoder_hidden)
        scores = self.v_a(torch.tanh(
            self.W_a(encoder_outputs) + self.U_a(decoder_hidden)
        )).squeeze(2)  # (batch, seq_len)

        # 计算注意力权重（softmax）
        attention_weights = torch.softmax(scores, dim=1).unsqueeze(1)  # (batch, 1, seq_len)

        # 计算上下文向量
        context_vector = torch.bmm(attention_weights, encoder_outputs).squeeze(1)  # (batch, encoder_hidden_dim)

        return context_vector, attention_weights.squeeze(1)

class Encoder(nn.Module):
    """
    编码器：将输入序列编码为上下文向量（双向LSTM）
    """

    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers=2, dropout_rate=0.0, bidirectional=True):
        """
        初始化编码器

        参数:
            input_size: 输入词汇表大小（英文字符数量）
            embedding_dim: 嵌入维度
            hidden_dim: LSTM 隐藏状态维度（每个方向）
            num_layers: LSTM层数
            dropout_rate: Dropout 率（层间dropout）
            bidirectional: 是否双向
        """
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 计算编码器输出维度
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * self.num_directions

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, seq_length)

        返回:
            outputs: LSTM 所有时间步的输出，形状为 (batch_size, seq_length, hidden_dim * num_directions)
            hidden_state: 最后时间步的隐藏状态，形状为 (num_layers * num_directions, batch_size, hidden_dim)
            cell_state: 最后时间步的细胞状态，形状为 (num_layers * num_directions, batch_size, hidden_dim)
        """
        # 嵌入层: (batch, seq_len) → (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)

        if self.dropout is not None:
            embedded = self.dropout(embedded)

        # LSTM 层: (batch, seq_len, embedding_dim) →
        # outputs: (batch, seq_len, hidden_dim * num_directions)
        # hidden_state, cell_state: (num_layers * num_directions, batch, hidden_dim)
        outputs, (hidden_state, cell_state) = self.lstm(embedded)

        # 对双向LSTM，取最后层的两个方向的隐藏状态拼接作为解码器初始状态
        # 但解码器只需要单向的隐藏状态，所以这里返回所有状态，解码器会处理
        return outputs, hidden_state, cell_state

class Decoder(nn.Module):
    """
    解码器：使用上下文向量生成目标序列（带注意力机制）
    """

    def __init__(self, output_size, embedding_dim, hidden_dim, encoder_hidden_dim=None, dropout_rate=0.0, num_layers=1, attention=None):
        """
        初始化解码器

        参数:
            output_size: 输出词汇表大小（中文字符数量）
            embedding_dim: 嵌入维度
            hidden_dim: LSTM 隐藏状态维度
            encoder_hidden_dim: 编码器输出维度（如果为None则等于hidden_dim）
            dropout_rate: Dropout 率
            num_layers: LSTM层数
            attention: 注意力机制实例（可选）
        """
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)

        # 编码器输出维度（用于注意力上下文向量维度）
        self.encoder_hidden_dim = encoder_hidden_dim if encoder_hidden_dim is not None else hidden_dim

        # LSTM输入维度：嵌入维度 + 上下文向量维度（如果有注意力）
        lstm_input_dim = embedding_dim + (self.encoder_hidden_dim if attention is not None else 0)

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 and num_layers > 1 else 0,
            bidirectional=False
        )

        # 全连接层输入维度：LSTM输出维度 + 上下文向量维度（如果有注意力）
        fc_input_dim = hidden_dim + (self.encoder_hidden_dim if attention is not None else 0)
        self.fc = nn.Linear(fc_input_dim, output_size)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.attention = attention

        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden_state, cell_state, encoder_outputs=None):
        """
        前向传播（单步解码）

        参数:
            x: 当前输入字符张量，形状为 (batch_size, 1)
            hidden_state: LSTM 隐藏状态，形状为 (num_layers, batch_size, hidden_dim)
            cell_state: LSTM 细胞状态，形状为 (num_layers, batch_size, hidden_dim)
            encoder_outputs: 编码器所有时间步的输出，形状为 (batch_size, seq_len, encoder_hidden_dim)

        返回:
            output: 下一个字符的预测概率分布，形状为 (batch_size, output_size)
            hidden_state: 更新后的隐藏状态
            cell_state: 更新后的细胞状态
        """
        # 嵌入层: (batch, 1) → (batch, 1, embedding_dim)
        embedded = self.embedding(x)

        if self.dropout is not None:
            embedded = self.dropout(embedded)

        # 如果有注意力机制，计算上下文向量
        context_vector = None
        if self.attention is not None and encoder_outputs is not None:
            # 解码器隐藏状态（需要从 (num_layers, batch, hidden_dim) 转换为 (batch, hidden_dim)）
            # 使用最后一层的隐藏状态
            decoder_hidden = hidden_state[-1, :, :]  # (batch, hidden_dim)
            context_vector, _ = self.attention(decoder_hidden, encoder_outputs)
            # 将上下文向量与嵌入向量拼接
            embedded = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)  # (batch, 1, embedding_dim+encoder_hidden_dim)

        # LSTM 层: (batch, 1, input_dim) → (batch, 1, hidden_dim)
        # hidden_state, cell_state: (num_layers, batch, hidden_dim)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded, (hidden_state, cell_state))

        # 如果有注意力机制，将LSTM输出与上下文向量拼接
        if self.attention is not None and encoder_outputs is not None:
            lstm_out = torch.cat((lstm_out, context_vector.unsqueeze(1)), dim=2)  # (batch, 1, hidden_dim+encoder_hidden_dim)

        # 全连接层: (batch, 1, fc_input_dim) → (batch, 1, output_size)
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

    def _convert_bidirectional_hidden(self, hidden, cell):
        """
        将双向LSTM的隐藏状态和细胞状态转换为单向解码器可用的状态

        参数:
            hidden: 双向LSTM的隐藏状态，形状为 (num_layers * num_directions, batch_size, hidden_dim)
            cell: 双向LSTM的细胞状态，形状为 (num_layers * num_directions, batch_size, hidden_dim)

        返回:
            hidden_unidir: 单向隐藏状态，形状为 (num_layers, batch_size, hidden_dim)
            cell_unidir: 单向细胞状态，形状为 (num_layers, batch_size, hidden_dim)
        """
        # 获取编码器的方向数
        num_directions = self.encoder.num_directions
        num_layers = self.encoder.lstm.num_layers

        # 如果是单向LSTM，直接返回
        if num_directions == 1:
            return hidden, cell

        # 将双向状态转换为单向：对每个层的两个方向状态求和
        # 隐藏状态形状: (num_layers * num_directions, batch, hidden_dim)
        batch_size = hidden.size(1)
        hidden_dim = hidden.size(2)

        # 重塑为 (num_layers, num_directions, batch, hidden_dim)
        hidden_reshaped = hidden.view(num_layers, num_directions, batch_size, hidden_dim)
        cell_reshaped = cell.view(num_layers, num_directions, batch_size, hidden_dim)

        # 对两个方向求和得到单向状态
        hidden_unidir = hidden_reshaped.sum(dim=1)  # (num_layers, batch, hidden_dim)
        cell_unidir = cell_reshaped.sum(dim=1)      # (num_layers, batch, hidden_dim)

        return hidden_unidir, cell_unidir

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
        encoder_outputs, hidden_state, cell_state = self.encoder(source)

        # 将双向LSTM状态转换为单向解码器可用状态
        hidden_state, cell_state = self._convert_bidirectional_hidden(hidden_state, cell_state)

        # 解码器的第一个输入是起始标记
        decoder_input = target[:, 0].unsqueeze(1)  # (batch, 1)

        # 逐步解码
        for t in range(1, target_len):
            # 解码器单步前向传播
            output, hidden_state, cell_state = self.decoder(
                decoder_input, hidden_state, cell_state, encoder_outputs)

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
        encoder_outputs, hidden_state, cell_state = self.encoder(source)

        # 将双向LSTM状态转换为单向解码器可用状态
        hidden_state, cell_state = self._convert_bidirectional_hidden(hidden_state, cell_state)

        # 解码器的第一个输入是起始标记
        decoder_input = torch.tensor([[start_token_idx]] * batch_size).to(self.device)  # (batch, 1)

        # 存储生成的序列
        outputs = torch.zeros(batch_size, max_length).long().to(self.device)

        # 逐步解码
        for t in range(max_length):
            # 解码器单步前向传播
            output, hidden_state, cell_state = self.decoder(
                decoder_input, hidden_state, cell_state, encoder_outputs)

            # 获取预测的下一个字符（概率最大的索引）
            top1 = output.argmax(1).unsqueeze(1)  # (batch, 1)

            # 存储预测结果
            outputs[:, t] = top1.squeeze(1)

            # 更新解码器输入
            decoder_input = top1

        return outputs

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, teacher_forcing_ratio, grad_clip=1.0):
    """
    训练一个epoch

    参数:
        model: 要训练的模型
        dataloader: 训练数据 DataLoader
        criterion: 损失函数
        optimizer: 优化器
        teacher_forcing_ratio: 教师强制比例
        grad_clip: 梯度裁剪阈值

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)  # 梯度裁剪
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
    print("\n构建英文词级词汇表...")
    input_token2idx, input_idx2token = build_vocab(input_texts, max_items=5000, tokenizer=tokenize_english_word_level)

    print("\n构建中文字符词汇表...")
    target_token2idx, target_idx2token = build_vocab(target_texts, max_items=2000)  # 保持字符级

    # 3. 创建数据集和数据加载器
    print("\n创建数据集...")
    dataset = TranslationDataset(
        input_texts, target_texts,
        input_token2idx, target_token2idx,
        MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
        input_tokenizer=tokenize_english_word_level,  # 英文使用词级分词器
        target_tokenizer=None  # 中文保持字符级
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
        input_size=len(input_token2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=3,
        dropout_rate=DROPOUT,
        bidirectional=True
    )

    # 创建注意力机制
    attention = Attention(encoder_hidden_dim=encoder.output_dim, decoder_hidden_dim=HIDDEN_DIM)

    decoder = Decoder(
        output_size=len(target_token2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        encoder_hidden_dim=encoder.output_dim,
        dropout_rate=DROPOUT,
        num_layers=3,  # 与编码器层数保持一致
        attention=attention
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

    # 应用权重初始化
    model.apply(init_weights)

    # 6. 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记

    # 优化器：AdamW（带权重衰减的Adam）
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # 学习率调度器：余弦退火热身
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=LEARNING_RATE * 0.01)

    # 梯度裁剪阈值
    GRAD_CLIP = 5.0

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
            model, train_loader, criterion, optimizer, TEACHER_FORCING_RATIO, GRAD_CLIP)

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

        # 更新学习率
        scheduler.step()

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

        # 将索引转换为标记
        # 英文（词级）：需要将词连接起来
        source_tokens = []
        for i in source[0].cpu().numpy():
            if i == 0:  # 填充标记
                continue
            if i < len(input_idx2token):
                source_tokens.append(input_idx2token[i])
        source_text = ' '.join(source_tokens)

        # 中文（字符级）：直接连接字符
        target_chars = []
        for i in target[0].cpu().numpy():
            if i in [0, 1, 2]:  # 忽略填充、起始、结束标记
                continue
            if i < len(target_idx2token):
                target_chars.append(target_idx2token[i])
        target_text = ''.join(target_chars)

        predicted_chars = []
        for i in prediction[0].cpu().numpy():
            if i in [0, 1, 2]:  # 忽略填充、起始、结束标记
                continue
            if i < len(target_idx2token):
                predicted_chars.append(target_idx2token[i])
        predicted_text = ''.join(predicted_chars)

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