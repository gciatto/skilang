import unittest
from skilang import *
from tests import *


class TestSkilangClasses(unittest.TestCase):
    maxDiff = None

    def _test_parse(self, input: str, expected: Formula):
        parsed = parse(input)
        self.assertEqual(parsed, expected)

    def test_parsing_comparison_operators(self):
        for op in comparison_operators.keys():
            with self.subTest(operator=op):
                self._test_parse(f'x {op} 1', Expression(op, Symbol('x'), Term(1)))

    def test_parsing_comparison_operators_term_first(self):
        for op, parsed in comparison_operators.items():
            expression = f'1 {op} x'
            expected = Expression(parsed, Symbol('x'), Term(1))
            with self.subTest(expression=expression, parsed_as=str(expected)):
                self._test_parse(expression, expected)

    def test_parse(self):
        for input, expected in expressions.items():
            with self.subTest(input=input):
                self._test_parse(input, expected)

    def test_repr(self):
        for expression in expressions.values():
            string = repr(expression)
            with self.subTest(expression=string):
                self.assertEqual(eval(string), expression)

    def test_str(self):
        for expression in expressions.values():
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

    def test_evaluate_arithmetic(self):
        for string, expression in single_operator_expressions.items():
            assignments = {'x': 1}
            with self.subTest(expression=string, when=assignments):
                expected = eval(string.replace('x', '1'))
                actual = expression.evaluate(**assignments)
                self.assertEqual(expected, actual)

    def test_evaluate_access(self):
        class X:
            def __init__(self):
                self.y = [1, 2, 3]
            
            def f(self, *args):
                return 42 + sum(args)
            
        def inc(x):
            return x + 1
            
        assignments = {'x': X(), 'a': 3, 'inc': inc}
        expressions = {
            'inc(a)': 4,
            'x.y[0]': 1,
            'x.y[1]': 2,
            'x.y[2]': 3,
            'x.f()': 42,
            'x.f(1)': 43,
            'x.f(1, 2)': 45,
            'x.f(1, 2, a)': 48,
            'x.f().to_bytes(1, "big")': b'*',
            'x.f(1).to_bytes(1, "big")': b'+',
            'x.f(1).to_bytes(1, "big")[0]': 43,
            'a == 3': True,
            'inc(a) != 4': False,
            'x.y[1] > 1': True,
            'x.y[1] < 2': False,
            'x.y[1] <= 2': True,
            'x.y[1] >= 2': True,
        }

        for string, expected in expressions.items():
            with self.subTest(expression=string, when=assignments):
                expression = parse(string)
                actual = expression.evaluate(**assignments)
                self.assertEqual(expected, actual)

    def test_evaluate_overriding_operator(self):
        def minus(a, b, **assignments):
            return a.evaluate(**assignments) - b.evaluate(**assignments)
        
        expr = parse('x + y')
        assignments = {'x': 1, 'y': 2, '+': minus}
        result = expr.evaluate(**assignments)
        self.assertEqual(result, -1)