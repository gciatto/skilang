import unittest
from skilang import *


expressions = {
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


expressions[r'(a + (b * c)) - (((d / e) // f) % d)'] = expressions[r'a + b * c - d / e // f % d']
expressions[r'((a + b) * (c - d)) / (e // (f % d))'] = expressions[r'(a+b)*(c-d)/(e//(f%d))']


class TestParsing(unittest.TestCase):
    maxDiff = None

    def _test_parse(self, input: str, expected: Formula):
        parsed = parse(input)
        self.assertEqual(parsed, expected)

    def test_parse(self):
        for input, expected in expressions.items():
            with self.subTest(input=input):
                self._test_parse(input, expected)
