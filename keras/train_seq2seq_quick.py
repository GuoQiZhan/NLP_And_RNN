#!/usr/bin/env python
"""
快速训练 Seq2Seq 模型，用于验证训练循环
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# 添加当前目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(__file__))
from Sequence_to_Sequence_pytorch import (
    load_translation_data,
    build_char_vocab,
    TranslationDataset,
    Encoder,
    Decoder,
    Seq2Seq,
    DEVICE,
    train_epoch,
    evaluate
)

def main():
    print("快速训练 Seq2Seq 模型")
    print("=" * 50)

    # 1. 加载少量数据
    data_path = os.path.join("..", "dataset", "cmn-eng", "cmn.txt")
    input_texts, target_texts = load_translation_data(data_path, max_samples=200)
    print(f"加载了 {len(input_texts)} 个样本")

    # 2. 构建词汇表
    print("\n构建词汇表...")
    input_char2idx, input_idx2char = build_char_vocab(input_texts, max_chars=100)
    target_char2idx, target_idx2char = build_char_vocab(target_texts, max_chars=500)

    print(f"英文字符词汇表大小: {len(input_char2idx)}")
    print(f"中文字符词汇表大小: {len(target_char2idx)}")

    # 3. 创建数据集
    print("\n创建数据集...")
    max_input_length = 30
    max_target_length = 30

    dataset = TranslationDataset(
        input_texts, target_texts,
        input_char2idx, target_char2idx,
        max_input_length, max_target_length
    )

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 4. 创建模型（使用较小的维度）
    print("\n创建模型...")
    encoder = Encoder(
        input_size=len(input_char2idx),
        embedding_dim=64,
        hidden_dim=128,
        dropout_rate=0.2
    )

    decoder = Decoder(
        output_size=len(target_char2idx),
        embedding_dim=64,
        hidden_dim=128,
        dropout_rate=0.2
    )

    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. 训练一个epoch
    print("\n训练一个epoch...")
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, teacher_forcing_ratio=0.5)

    print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

    # 7. 验证
    print("\n验证...")
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

    # 8. 示例翻译
    print("\n示例翻译:")
    print("-" * 40)

    model.eval()
    sample_indices = list(range(min(3, len(val_dataset))))

    for idx in sample_indices:
        source, target = val_dataset[idx]
        source = source.unsqueeze(0).to(DEVICE)
        target = target.unsqueeze(0).to(DEVICE)

        # 获取预测
        prediction = model.predict(source, max_length=max_target_length+2)

        # 将索引转换为字符
        source_text = ''.join([input_idx2char[i] for i in source[0].cpu().numpy() if i != 0])
        target_text = ''.join([target_idx2char[i] for i in target[0].cpu().numpy() if i not in [0, 1, 2]])
        predicted_text = ''.join([target_idx2char[i] for i in prediction[0].cpu().numpy() if i not in [0, 1, 2]])

        # 打印结果
        print(f"英文: {source_text}")
        print(f"中文 (真实): {target_text}")
        print(f"中文 (预测): {predicted_text}")
        print("-" * 40)

    print("\n快速训练完成!")

if __name__ == "__main__":
    main()