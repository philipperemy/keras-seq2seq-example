import pickle


def build_vocabulary():
    vocabulary = set()
    with open('addresses2.txt', 'rb') as r:
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


if __name__ == '__main__':
    build_vocabulary()
