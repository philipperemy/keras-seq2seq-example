from random import randint, choice

import numpy as np

from utils import get_chars_and_ctable

VOCABULARY, _ = get_chars_and_ctable()


def add_char_randomly(input_str, vocabulary='abcdefghijklmnopqrstuvwxy'):
    random_char = choice(list(vocabulary))
    i = randint(a=0, b=len(input_str) - 1)
    output_str = input_str[0:i] + random_char + input_str[i:]
    assert len(output_str) == len(input_str) + 1
    assert random_char in output_str
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


def change_char_randomly(input_str, vocabulary='abcdefghijklmnopqrstuvwxy'):
    random_char = choice(list(vocabulary))
    i = randint(a=0, b=len(input_str) - 1)
    output_str = input_str[0:i] + random_char + input_str[i + 1:]
    assert len(output_str) == len(input_str)
    assert random_char in output_str
    return output_str


def permute_two_chars_randomly(input_str):
    def _swap(s, i, j):
        c = list(s)
        c[i], c[j] = c[j], c[i]
        return ''.join(c)

    # interest is limited here.
    i1 = randint(a=0, b=len(input_str) - 1)
    i2 = randint(a=0, b=len(input_str) - 1)
    output_str = _swap(input_str, i1, i2)
    return output_str


def add_data_augmentation(input_str, probs, vocabulary):
    nothing, add, remove, change = 0, 1, 2, 3
    status = np.random.choice(a=[nothing, add, remove, change], size=1, replace=False, p=probs)
    if status == nothing:
        return input_str, nothing
    elif status == add:
        return add_char_randomly(input_str, vocabulary), add
    elif status == remove:
        return remove_char_randomly(input_str), remove
    elif status == change:
        return change_char_randomly(input_str, vocabulary), change
    else:
        raise Exception('Invalid return.')


if __name__ == '__main__':
    while True:
        print(add_data_augmentation('hello world', [0.4, 0.2, 0.2, 0.2], VOCABULARY))
