"""
测试改进后的Seq2Seq模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# 将当前目录添加到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要的组件
from Sequence_to_Sequence_pytorch import (
    Encoder, Decoder, Seq2Seq, Attention,
    init_weights, DEVICE
)

def test_model_forward():
    """测试模型前向传播"""
    print("测试改进后的Seq2Seq模型...")

    # 模拟参数
    input_vocab_size = 1000
    output_vocab_size = 2000
    batch_size = 4
    src_len = 10
    tgt_len = 15
    embedding_dim = 256
    hidden_dim = 512

    # 创建模型组件
    encoder = Encoder(
        input_size=input_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout_rate=0.3,
        bidirectional=True
    )

    attention = Attention(
        encoder_hidden_dim=encoder.output_dim,
        decoder_hidden_dim=hidden_dim
    )

    decoder = Decoder(
        output_size=output_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        encoder_hidden_dim=encoder.output_dim,
        dropout_rate=0.3,
        num_layers=3,
        attention=attention
    )

    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # 应用权重初始化
    model.apply(init_weights)

    # 创建模拟数据
    source = torch.randint(0, input_vocab_size, (batch_size, src_len)).to(DEVICE)
    target = torch.randint(0, output_vocab_size, (batch_size, tgt_len)).to(DEVICE)

    print(f"输入形状: {source.shape}")
    print(f"目标形状: {target.shape}")

    # 测试训练模式
    print("\n测试训练模式...")
    model.train()
    output_train = model(source, target, teacher_forcing_ratio=0.5)
    print(f"训练输出形状: {output_train.shape}")
    print(f"期望形状: (batch_size={batch_size}, tgt_len-1={tgt_len-1}, vocab_size={output_vocab_size})")

    # 测试预测模式
    print("\n测试预测模式...")
    model.eval()
    with torch.no_grad():
        output_predict = model.predict(source, max_length=20, start_token_idx=1)
    print(f"预测输出形状: {output_predict.shape}")
    print(f"期望形状: (batch_size={batch_size}, max_length=20)")

    # 检查形状是否正确
    assert output_train.shape == (batch_size, tgt_len-1, output_vocab_size), \
        f"训练输出形状错误: {output_train.shape} != {(batch_size, tgt_len-1, output_vocab_size)}"

    assert output_predict.shape == (batch_size, 20), \
        f"预测输出形状错误: {output_predict.shape} != {(batch_size, 20)}"

    print("\n✅ 模型前向传播测试通过！")
    print(f"编码器输出维度: {encoder.output_dim}")
    print(f"编码器方向数: {encoder.num_directions}")
    print(f"编码器LSTM层数: {encoder.lstm.num_layers}")
    print(f"解码器LSTM层数: {decoder.num_layers}")

    # 清理
    del model, encoder, decoder, attention
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return True

def test_bidirectional_conversion():
    """测试双向LSTM隐藏状态转换"""
    print("\n测试双向LSTM隐藏状态转换...")

    # 模拟参数
    batch_size = 4
    hidden_dim = 512
    num_layers = 3
    num_directions = 2

    # 创建模拟隐藏状态
    hidden = torch.randn(num_layers * num_directions, batch_size, hidden_dim)
    cell = torch.randn(num_layers * num_directions, batch_size, hidden_dim)

    print(f"原始隐藏状态形状: {hidden.shape}")
    print(f"原始细胞状态形状: {cell.shape}")

    # 创建编码器实例（用于测试转换函数）
    encoder = Encoder(
        input_size=1000,
        embedding_dim=256,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=0.3,
        bidirectional=True
    )

    decoder = Decoder(
        output_size=2000,
        embedding_dim=256,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=0.3,
        attention=None
    )

    model = Seq2Seq(encoder, decoder, DEVICE)

    # 测试转换
    hidden_unidir, cell_unidir = model._convert_bidirectional_hidden(hidden, cell)

    print(f"转换后隐藏状态形状: {hidden_unidir.shape}")
    print(f"转换后细胞状态形状: {cell_unidir.shape}")

    assert hidden_unidir.shape == (num_layers, batch_size, hidden_dim), \
        f"转换后隐藏状态形状错误: {hidden_unidir.shape} != {(num_layers, batch_size, hidden_dim)}"

    assert cell_unidir.shape == (num_layers, batch_size, hidden_dim), \
        f"转换后细胞状态形状错误: {cell_unidir.shape} != {(num_layers, batch_size, hidden_dim)}"

    print("✅ 双向LSTM隐藏状态转换测试通过！")

    return True

if __name__ == "__main__":
    try:
        test_model_forward()
        test_bidirectional_conversion()
        print("\n🎉 所有测试通过！模型改进成功。")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)