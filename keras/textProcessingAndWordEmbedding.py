from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras import optimizers
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

VOCABULARY = 10000
EMBEDDING_DIM = 8
WORD_NUM = 20
EPOCHS = 50

print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCABULARY)

x_train = pad_sequences(x_train, maxlen=WORD_NUM)
x_test = pad_sequences(x_test, maxlen=WORD_NUM)

x_valid = x_test[:5000]
y_valid = y_test[:5000]
x_test = x_test[5000:]
y_test = y_test[5000:]

model = Sequential()
model.add(Embedding(VOCABULARY, EMBEDDING_DIM, input_length=WORD_NUM))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 显示模型摘要（在训练前）
print("Model summary:")
model.build(input_shape=(None, WORD_NUM))
print(model.summary())

# 训练模型
print("Starting training...")
history = model.fit(x_train, y_train, epochs=EPOCHS,
                    batch_size=32, validation_data=(x_valid, y_valid))

# 评估模型
print("Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

loss_and_acc = model.evaluate(x_test, y_test)
print('loss = ' + str(loss_and_acc[0]))
print('acc = ' + str(loss_and_acc[1]))
