import numpy as np
from keras.layers import Dense, Flatten, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

from helpers import batch
from load_data import get_batch, VOCAB_SIZE, TOKEN_INDICES

# <class 'tuple'>: (10, 9)
output_dim = 9
x, y = get_batch()
xt = np.transpose(np.expand_dims(batch(x)[0], axis=2), (1, 0, 2))
yt = np.reshape(to_categorical(np.array(y).flatten(), VOCAB_SIZE), (-1, output_dim, VOCAB_SIZE))

# yt.argmax(axis=2) == y

time_len = 19
input_dim = 1

m = Sequential()
m.add(Flatten(input_shape=(time_len, input_dim)))
m.add(Dense(output_dim * VOCAB_SIZE))
m.add(Reshape((output_dim, VOCAB_SIZE)))
m.add(TimeDistributed(Dense(VOCAB_SIZE, activation='softmax')))
# m.add(LSTM(32, input_shape=(time_len, input_dim)))
# m.add(LSTM(32, input_shape=(time_len, input_dim)))
# m.add(Dense(32))
# m.add(RepeatVector(output_dim))
# m.add(TimeDistributed(Dense(VOCAB_SIZE, activation='softmax')))

from keras.optimizers import Adam
adam = Adam(lr=0.0001)
m.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

m.fit(xt, yt, epochs=100000)

print(np.reshape([TOKEN_INDICES[a] for a in list(m.predict(xt).argmax(axis=2).flatten())], (-1, output_dim)))
