# -*- coding: utf-8 -*-


import os

import numpy as np
from keras import layers
from keras.models import Sequential
from six.moves import range

from constants import TOKEN_INDICES
from data_gen import LazyDataLoader


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


DATA_LOADER = LazyDataLoader()

MAXLEN, OUTPUT_MAX_LEN, TRAINING_SIZE = DATA_LOADER.statistics()

# Parameters for the model and dataset.
# TRAINING_SIZE = 10
# OUTPUT_MAX_LEN = 9
INVERT = True

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
# MAXLEN = 19

chars = ''.join(list(TOKEN_INDICES.values()))
ctable = CharacterTable(chars)

questions = []
expected = []
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    x, y = DATA_LOADER.next()
    # Pad the data with spaces such that it is always MAXLEN.
    q = x
    query = q
    ans = y
    if INVERT:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
        # space used for padding.)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

if not os.path.exists('x_y.npz'):
    print('Vectorization...')

    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(questions), OUTPUT_MAX_LEN, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        print(i)
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        print(i)
        y[i] = ctable.encode(sentence, OUTPUT_MAX_LEN)

    # Shuffle (x, y) in unison as the later parts of x will almost all be larger
    # digits.
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    np.savez_compressed('x_y.npz', x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)

else:
    print('Loading prefetched data...')
    data = np.load('x_y.npz')
    x_train = data['x_train']
    x_val = data['x_val']
    y_train = data['y_train']
    y_val = data['y_val']

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 256
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')


def model_1():
    m = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    m.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    # As the decoder RNN's input, repeatedly provide with the last hidden state of
    # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    m.add(layers.RepeatVector(OUTPUT_MAX_LEN))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(LAYERS):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        m.add(RNN(HIDDEN_SIZE, return_sequences=True))
    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    m.add(layers.TimeDistributed(layers.Dense(len(chars))))
    m.add(layers.Activation('softmax'))
    return m


def model_2():
    # too big in Memory!
    m = Sequential()
    from keras.layers.core import Flatten, Dense, Reshape
    from keras.layers.wrappers import TimeDistributed
    m.add(Flatten(input_shape=(MAXLEN, len(chars))))
    m.add(Dense(OUTPUT_MAX_LEN * len(chars)))
    m.add(Reshape((OUTPUT_MAX_LEN, len(chars))))
    m.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    return m


def model_3():
    m = Sequential()
    from keras.layers.core import Dense, Reshape
    from keras.layers.wrappers import TimeDistributed
    m.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    m.add(Dense(OUTPUT_MAX_LEN * len(chars)))
    m.add(Reshape((OUTPUT_MAX_LEN, len(chars))))
    m.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    return m


model = model_3()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=10,
              validation_data=(x_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_train))
        rowx, rowy = x_train[np.array([ind])], y_train[np.array([ind])]  # replace by x_val, y_val
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=" ")
        else:
            print(colors.fail + '☒' + colors.close, end=" ")
        print(guess)
        print('---')
