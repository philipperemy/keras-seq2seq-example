from random import randint, choice

from utils import get_chars_and_ctable

VOCABULARY, _ = get_chars_and_ctable()


def _swap(s, i, j):
    c = list(s)
    c[i], c[j] = c[j], c[i]
    return ''.join(c)


def change_char_randomly(input_str, vocabulary='abcdefghijklmnopqrstuvwxy'):
    random_char = choice(list(vocabulary))
    i = randint(a=0, b=len(input_str) - 1)
    output_str = input_str[0:i] + random_char + input_str[i + 1:]
    assert len(output_str) == len(input_str)
    assert random_char in output_str
    return output_str


def permute_two_chars_randomly(input_str):
    # interest is limited here.
    i1 = randint(a=0, b=len(input_str) - 1)
    i2 = randint(a=0, b=len(input_str) - 1)
    output_str = _swap(input_str, i1, i2)
    return output_str


def remove_char_randomly(input_str):
    i = randint(a=0, b=len(input_str) - 1)
    char_to_remove = input_str[i]
    before = input_str.count(char_to_remove)
    output_str = input_str[0:i] + input_str[i + 1:]
    after = output_str.count(char_to_remove)
    assert before - 1 == after
    assert len(output_str) == len(input_str) - 1
    return output_str


while True:
    print(change_char_randomly('石川県小松市荒木田町ヘ１３８−１', VOCABULARY))
    # permute_two_chars_randomly('石川県小松市荒木田町ヘ１３８−１')
    print(remove_char_randomly('石川県小松市荒木田町ヘ１３８−１'))
