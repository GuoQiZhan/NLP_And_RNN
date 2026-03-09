#!/usr/bin/env python
"""
快速测试 Seq2Seq 模型的数据加载和模型构建
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from Sequence_to_Sequence_pytorch import (
    load_translation_data,
    build_char_vocab,
    TranslationDataset,
    Encoder,
    Decoder,
    Seq2Seq,
    DEVICE
)

def test_data_loading():
    """测试数据加载"""
    print("测试数据加载...")
    data_path = os.path.join("..", "dataset", "cmn-eng", "cmn.txt")
    input_texts, target_texts = load_translation_data(data_path, max_samples=100)
    print(f"加载了 {len(input_texts)} 个样本")
    print(f"示例英文: {input_texts[0]}")
    print(f"示例中文: {target_texts[0]}")
    return input_texts, target_texts

def test_vocab_building(input_texts, target_texts):
    """测试词汇表构建"""
    print("\n测试词汇表构建...")
    input_char2idx, input_idx2char = build_char_vocab(input_texts, max_chars=50)
    target_char2idx, target_idx2char = build_char_vocab(target_texts, max_chars=100)

    print(f"英文字符词汇表大小: {len(input_char2idx)}")
    print(f"中文字符词汇表大小: {len(target_char2idx)}")
    print(f"英文前10个字符: {list(input_char2idx.keys())[:10]}")
    print(f"中文前10个字符: {list(target_char2idx.keys())[:10]}")

    return input_char2idx, target_char2idx

def test_dataset(input_texts, target_texts, input_char2idx, target_char2idx):
    """测试数据集"""
    print("\n测试数据集...")
    dataset = TranslationDataset(
        input_texts, target_texts,
        input_char2idx, target_char2idx,
        max_input_length=20,
        max_target_length=20
    )

    print(f"数据集大小: {len(dataset)}")
    source, target = dataset[0]
    print(f"输入张量形状: {source.shape}")
    print(f"目标张量形状: {target.shape}")
    print(f"输入序列: {source.tolist()}")
    print(f"目标序列: {target.tolist()}")

    return dataset

def test_model_building(input_char2idx, target_char2idx):
    """测试模型构建"""
    print("\n测试模型构建...")
    import torch

    encoder = Encoder(
        input_size=len(input_char2idx),
        embedding_dim=32,
        hidden_dim=64,
        dropout_rate=0.1
    )

    decoder = Decoder(
        output_size=len(target_char2idx),
        embedding_dim=32,
        hidden_dim=64,
        dropout_rate=0.1
    )

    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    print("模型构建成功!")
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters())}")
    print(f"解码器参数数量: {sum(p.numel() for p in decoder.parameters())}")

    # 测试前向传播
    batch_size = 2
    source_len = 10
    target_len = 12

    source = torch.randint(1, len(input_char2idx), (batch_size, source_len))
    target = torch.randint(1, len(target_char2idx), (batch_size, target_len))

    source = source.to(DEVICE)
    target = target.to(DEVICE)

    # 训练模式
    output = model(source, target, teacher_forcing_ratio=0.5)
    print(f"输出形状: {output.shape}")  # 应为 (batch_size, target_len-1, vocab_size)

    # 预测模式
    prediction = model.predict(source, max_length=15)
    print(f"预测形状: {prediction.shape}")  # 应为 (batch_size, max_length)

    return model

def main():
    """主测试函数"""
    print("Seq2Seq 模型快速测试")
    print("=" * 50)

    # 测试数据加载
    input_texts, target_texts = test_data_loading()

    # 测试词汇表构建
    input_char2idx, target_char2idx = test_vocab_building(input_texts, target_texts)

    # 测试数据集
    dataset = test_dataset(input_texts, target_texts, input_char2idx, target_char2idx)

    # 测试模型构建
    import torch
    model = test_model_building(input_char2idx, target_char2idx)

    print("\n所有测试通过!")

if __name__ == "__main__":
    main()