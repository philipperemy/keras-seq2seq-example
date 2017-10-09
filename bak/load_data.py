import pickle

import numpy as np

MAX_LEN = 30

# list(map(chr, tokens)) chr(ord(x)) = x


# Token -> Index: <function at ORD>
# Index -> Token: <function at CHR>

INDICES_TOKEN = pickle.load(open('indices_token.pkl', 'rb'))
TOKEN_INDICES = pickle.load(open('token_indices.pkl', 'rb'))

VOCAB_SIZE = len(TOKEN_INDICES)


def str_to_tokens(string):
    return list(map(lambda kk: INDICES_TOKEN[kk], list(string)))
    # return list(map(lambda kk: ord(kk), list(string)))


def str_to_tokens_pad(string, max_len=MAX_LEN):
    tokens = str_to_tokens(string)
    tokens = tokens[0:max_len]
    tokens_len = len(tokens)
    tokens.extend((max_len - len(tokens)) * [0])
    # tokens = tokens[::-1]  # flip because it's more efficient for seq2seq.
    print(tokens, len(tokens))
    return tokens, tokens_len


def build_vocabulary():
    vocabulary = set()
    with open('addresses.txt', 'rb') as r:
        for l in r.readlines():
            y, x = l.decode('utf8').strip().split('　')
            for element in list(y):
                vocabulary.add(element)
            for element in list(x):
                vocabulary.add(element)
    vocabulary = sorted(list(vocabulary))
    print(vocabulary)
    token_indices = dict((c, i) for (c, i) in enumerate(vocabulary))
    indices_token = dict((i, c) for (c, i) in enumerate(vocabulary))

    with open('token_indices.pkl', 'wb') as w:
        pickle.dump(obj=token_indices, file=w)

    with open('indices_token.pkl', 'wb') as w:
        pickle.dump(obj=indices_token, file=w)


# build_vocabulary()


def get_partial():
    full_x = []
    full_y = []
    with open('addresses.txt', 'rb') as r:
        for l in r.readlines():
            y, x = l.decode('utf8').strip().split('　')
            tokens_x = str_to_tokens(x)
            tokens_y = str_to_tokens(y)

            full_x.append(tokens_x)
            full_y.append(tokens_y)

    return full_x, full_y


def get_raw():
    x_list = []
    y_list = []
    with open('addresses.txt', 'rb') as r:
        for l in r.readlines():
            y, x = l.decode('utf8').strip().split('　')
            x_list.append(x)
            y_list.append(y)
    return x_list, y_list


def get_batch(bs=2):
    return get_partial()


def get():
    full_x = []
    full_y = []
    full_l = []
    vocab = set()
    with open('addresses.txt', 'rb') as r:
        for l in r.readlines():
            y, x = l.decode('utf8').strip().split('　')
            tokens_x, tokens_x_len = str_to_tokens_pad(x)
            tokens_y = str_to_tokens(y)

            for token in tokens_x:
                vocab.add(token)

            for token in tokens_y:
                vocab.add(token)

            full_x.append(tokens_x)
            full_y.append(tokens_y)
            full_l.append(tokens_x_len)

    vocab_size = len(vocab)
    full_x = np.array(full_x)
    full_y = np.array(full_y)
    full_l = np.array(full_l)
    return full_x, full_y, full_l, vocab_size


if __name__ == '__main__':
    get()

#
#
# self.chars = sorted(set(chars))
# self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
# self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

# print(ord('山'))
# print(ord('の'))
# print(ord('手'))
