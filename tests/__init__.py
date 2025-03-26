from itertools import permutations


def all_permutations(iterable):
    items = list(iterable)
    for i in range(len(items) + 1):
        for permutation in permutations(items, i):
            yield permutation
    