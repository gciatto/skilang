from itertools import permutations
from skilang import *


def all_permutations(iterable):
    items = list(iterable)
    for i in range(len(items) + 1):
        for permutation in permutations(items, i):
            yield permutation


meaningless_yet_valid_expressions = {
    r'a + b * c - d / e // f % d': 
        Expression('-', Expression('+', Symbol('a'), Expression('*', Symbol('b'), Symbol('c'))), Expression('%', Expression('//', Expression('/', Symbol('d'), Symbol('e')), Symbol('f')), Symbol('d'))),
    r'(a+b)*(c-d)/(e//(f%d))' : 
        Expression('/', Expression('*', Expression('+', Symbol('a'), Symbol('b')), Expression('-', Symbol('c'), Symbol('d'))), Expression('//', Symbol('e'), Expression('%', Symbol('f'), Symbol('d')))),
    f'x.y[0]': 
        Expression('[]', Expression('.', Symbol('x'), Symbol('y')), Term(0)),
    f'x.y[0].z': 
        Expression('.', Expression('[]', Expression('.', Symbol('x'), Symbol('y')), Term(0)), Symbol('z')),
    f'+x & -y | ~z': 
        Expression('|', Expression('&', Expression('+', Symbol('x')), Expression('-', Symbol('y'))), Expression('~', Symbol('z'))),
    f'f() + g(x) + h(y, z)':
        Expression('+', Expression('+', Predicate('f', ), Predicate('g', Symbol('x'))), Predicate('h', Symbol('y'), Symbol('z'))),
    f'x.f(1)':
        Expression('.', Symbol('x'), Predicate('f', Term(1))),
}

single_operator_expressions = {}

binary_operators = list("+-*/%&|^") + ['//', '**', '<<', '>>']
unary_operators = list("~+-")

for op in binary_operators:
    single_operator_expressions[f'x {op} 1'] = Expression(op, Symbol('x'), Term(1))
    single_operator_expressions[f'1 {op} x'] = Expression(op, Term(1), Symbol('x'))

for op in unary_operators:
    single_operator_expressions[f'{op}x'] = Expression(op, Symbol('x'))


expressions = dict(**meaningless_yet_valid_expressions, **single_operator_expressions)


def generate_formulas():
    symbols = tuple(Symbol(s) for s in 'xy')
    yield from symbols
    terms = tuple(Term(i) for i in range(1))
    yield from terms
    predicates = tuple(Predicate(f, *args) for f in "fg" for args in all_permutations(symbols + terms))
    yield from predicates
    expressions = tuple(Expression(op, *args) for op in '+-' for args in permutations(symbols + terms + predicates, 2))
    yield from expressions