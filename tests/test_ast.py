import unittest
from skilang import *
from tests import all_permutations, permutations


valid_expressions = {
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
}


def generate_formulas():
    symbols = tuple(Symbol(s) for s in 'xy')
    yield from symbols
    terms = tuple(Term(i) for i in range(1))
    yield from terms
    predicates = tuple(Predicate(f, *args) for f in "fg" for args in all_permutations(symbols + terms))
    yield from predicates
    expressions = tuple(Expression(op, *args) for op in '+-' for args in permutations(symbols + terms + predicates, 2))
    yield from expressions


class TestSkilangClasses(unittest.TestCase):
    maxDiff = None

    def _test_parse(self, input: str, expected: Formula):
        parsed = parse(input)
        self.assertEqual(parsed, expected)

    def test_parse(self):
        for input, expected in valid_expressions.items():
            with self.subTest(input=input):
                self._test_parse(input, expected)

    def test_repr(self):
        for expression in valid_expressions.values():
            string = repr(expression)
            with self.subTest(expression=string):
                self.assertEqual(eval(string), expression)

    def test_str(self):
        for expression in valid_expressions.values():
            string = str(expression)
            with self.subTest(expression=string):
                self.assertEqual(parse(string), expression)

    def test_hash(self):
        formulas = list(generate_formulas())
        set_of_formulas = set(formulas)
        for formula in formulas:
            self.assertIn(formula, set_of_formulas)

    def test_sort(self):
        import random

        formulas = list(generate_formulas())
        ordered = sorted(formulas)
        shuffled = list(formulas)
        random.shuffle(shuffled)
        shuffled.sort()
        self.assertEqual(ordered, shuffled)
