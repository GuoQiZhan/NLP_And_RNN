from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Flatten
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers

VOCABULARY = 10000
EMBEDDING_DIM = 32
WORD_NUM = 500
STATE_DIM = 32
EPOCHS = 10

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
model.add(LSTM(STATE_DIM, return_sequences=True, dropout=0.2))
model.add(LSTM(STATE_DIM, return_sequences=True, dropout=0.2))
model.add(LSTM(STATE_DIM, return_sequences=False, dropout=0.2))
model.add(Dense(1, activation = 'sigmoid'))


# Before Training
print("Before Training:")
model.build(input_shape=(None, WORD_NUM))
print(model.summary())

# Training
print("Training model...")
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=EPOCHS,
                    batch_size=32, validation_data=(x_valid, y_valid))

# Evaluation
print("Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

loss_and_acc = model.evaluate(x_test, y_test)
print('loss = ' + str(loss_and_acc[0]))
print('acc = ' + str(loss_and_acc[1]))
