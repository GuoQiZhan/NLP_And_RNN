"""
SimpleRNN 模型在 IMDB 情感分析任务上的实现

模型架构说明：
model.add(SimpleRNN(units, return_sequences=True))
    若 return_sequences=True，则输出每个时间步的隐藏状态
    若 return_sequences=False，则输出最后一个时间步的隐藏状态

    True :  改为True时 需要添加个Flatten层将输出展平为一维向量 Accuracy = 0.85
    False:  Accuracy = 0.85

SimpleRNN 的缺点：
    1. 梯度消失问题：在处理长序列时 SimpleRNN 容易遇到梯度消失问题 导致模型无法学习长距离依赖关系。
    2. 计算复杂度高：SimpleRNN 是递归神经网络 每个时间步的计算都依赖于前一个时间步的隐藏状态 计算复杂度较高。
    3. 只擅长处理短序列 记忆力只有七秒 会遗忘很久之前的信息

文件结构：
    1. 导入必要的库
    2. 定义超参数
    3. 加载和预处理 IMDB 数据集
    4. 构建 SimpleRNN 模型
    5. 编译和训练模型
    6. 评估模型性能
"""

# ==================== 导入必要的库 ====================
# 导入 Keras 模型和层
from keras.models import Sequential  # 顺序模型，用于按顺序堆叠神经网络层
from keras.layers import SimpleRNN, Embedding, Dense, Flatten  # 导入神经网络层：SimpleRNN、嵌入层、全连接层、展平层
# 导入 IMDB 数据集
from keras.datasets import imdb  # IMDB 电影评论数据集，用于情感分析任务
# 导入序列处理工具
from keras.preprocessing.sequence import pad_sequences  # 用于填充序列到相同长度
# 导入优化器
from keras import optimizers  # 包含各种优化算法（如 RMSprop、Adam 等）

# ==================== 定义超参数 ====================
'''
超参数说明：
vocabulary: unique words in the dictionary          词典里有10000个单词
embedding_dim: shape(x) = 32                        每个单词用32维向量表示
word_num: sequence length                           每个样本的序列长度为500
state_dim : shape(h) = 32                           隐藏层的维度为32
'''

VOCABULARY = 10000       # 词汇表大小，只保留数据集中最常见的10000个单词
EMBEDDING_DIM = 32       # 词嵌入维度，每个单词用一个32维的向量表示
WORD_NUM = 500           # 序列长度，每个评论被填充或截断为500个单词
STATE_DIM = 32           # SimpleRNN 隐藏状态的维度，即隐藏层有32个神经元
EPOCHS = 30              # 训练轮数，整个数据集将被迭代30次

# ==================== 数据加载和预处理 ====================
print("Loading IMDB dataset...")  # 打印加载数据集的提示信息

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

# ==================== 构建 SimpleRNN 模型 ====================
# 创建一个顺序模型，层将按顺序堆叠
model = Sequential()

# 添加嵌入层（Embedding Layer）：
# 将整数索引（单词ID）转换为密集向量表示
# 参数：VOCABULARY=10000（输入维度），EMBEDDING_DIM=32（输出维度），input_length=WORD_NUM=500（输入序列长度）
model.add(Embedding(VOCABULARY, EMBEDDING_DIM, input_length=WORD_NUM))

# 添加 SimpleRNN 层：
# 处理序列数据，返回每个时间步的隐藏状态（return_sequences=True）
# 参数：STATE_DIM=32（隐藏状态的维度，即RNN单元的数量）
# return_sequences=True 表示返回完整序列的输出，而不是仅最后时间步的输出
model.add(SimpleRNN(STATE_DIM, return_sequences=True))  # 注意：设置为True时需要添加Flatten层

# 添加展平层（Flatten Layer）：
# 将RNN输出的3D张量（批量大小, 时间步数, 隐藏维度）展平为2D张量（批量大小, 时间步数*隐藏维度）
# 因为return_sequences=True时，每个时间步都有输出，需要展平才能输入到全连接层
model.add(Flatten())

# 添加全连接输出层（Dense Layer）：
# 单个神经元，使用sigmoid激活函数，输出0到1之间的值，表示正面评论的概率
model.add(Dense(1, activation='sigmoid'))

# ==================== 训练前准备 ====================
# 构建模型并指定输入形状（批量大小自动为None，序列长度为WORD_NUM=500）
model.build(input_shape=(None, WORD_NUM))

# 打印模型摘要，显示各层的参数数量和总参数数
print(model.summary())

# ==================== 模型编译和训练 ====================
print("Training model...")  # 打印开始训练提示信息

# 编译模型：配置学习过程
# optimizer: 使用RMSprop优化器，学习率设置为0.001
# loss: 使用二元交叉熵损失函数，适用于二分类问题
# metrics: 监控准确率指标
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型：
# x_train, y_train: 训练数据和标签
# epochs=EPOCHS=30: 训练轮数
# batch_size=32: 每个批次的样本数量
# validation_data=(x_valid, y_valid): 验证集，用于监控验证性能
# 返回history对象，包含训练过程中的损失和指标值
history = model.fit(x_train, y_train, epochs=EPOCHS,
                    batch_size=32, validation_data=(x_valid, y_valid))

# ==================== 模型评估 ====================
print("Evaluating model...")  # 打印开始评估提示信息

# 在测试集上评估模型性能
# evaluate方法返回损失值和指标值（这里是指定的准确率）
test_loss, test_acc = model.evaluate(x_test, y_test)

# 打印测试准确率，保留4位小数
print(f"Test accuracy: {test_acc:.4f}")

# 再次评估模型（冗余代码，与前一行功能相同）
# 获取损失和准确率
loss_and_acc = model.evaluate(x_test, y_test)

# 分别打印损失和准确率
print('loss = ' + str(loss_and_acc[0]))
print('acc = ' + str(loss_and_acc[1]))
