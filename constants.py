import pickle

MAX_LEN = 30

# list(map(chr, tokens)) chr(ord(x)) = x


# Token -> Index: <function at ORD>
# Index -> Token: <function at CHR>

INDICES_TOKEN = pickle.load(open('indices_token.pkl', 'rb'))
TOKEN_INDICES = pickle.load(open('token_indices.pkl', 'rb'))

VOCAB_SIZE = len(TOKEN_INDICES)


