from typing import Dict, Iterable

from skilang.utils import logger


class EvaluationError(Exception):
    def __init__(self, formula, message: str = None, assignments: dict = None):
        msg = f"Error evaluating formula {formula}"
        if assignments:
            msg += f" with assignments {assignments}"
        if message:
            msg += f": {message}"
        super().__init__(msg)


class Formula:
    __comparison_operators_work_as_python_expects = True

    @classmethod
    def _comparison_operators_as_builders(cls):
        def activate():
            cls.__comparison_operators_work_as_python_expects = False

        def deactivate():
            cls.__comparison_operators_work_as_python_expects = True

        class Context:
            def __enter__(self):
                activate()
                return cls

            def __exit__(self, exc_type, exc_value, traceback):
                deactivate()

        return Context()

    def evaluate(self, **kwargs):
        try:
            return self._evaluate(**kwargs)
        except EvaluationError:
            raise
        except Exception as e:
            raise EvaluationError(self, assignments=kwargs) from e

    def _evaluate(self, **kwargs):
        raise NotImplementedError

    def _force_formula(self, value, symbol: bool = False) -> "Formula":
        if isinstance(value, Formula):
            return value
        if symbol:
            return Symbol(value)
        return Term(value)

    def _check_equality(self, other) -> bool:
        raise NotImplementedError

    def _attributes_to_hash(self):
        raise NotImplementedError

    def _compare(self, other) -> int:
        raise NotImplementedError

    def __eq__(self, other):
        if self.__comparison_operators_work_as_python_expects:
            return self._check_equality(other)
        return self.__binary_expression("==", other)

    def __hash__(self):
        return hash(tuple(self._attributes_to_hash()))

    def __ne__(self, other):
        if self.__comparison_operators_work_as_python_expects:
            return not self._check_equality(other)
        return self.__binary_expression("!=", other)

    def __lt__(self, other):
        if self.__comparison_operators_work_as_python_expects:
            return self._compare(other) < 0
        return self.__binary_expression("<", other)

    def __le__(self, other):
        if self.__comparison_operators_work_as_python_expects:
            return self._compare(other) <= 0
        return self.__binary_expression("<=", other)

    def __gt__(self, other):
        if self.__comparison_operators_work_as_python_expects:
            return self._compare(other) > 0
        return self.__binary_expression(">", other)

    def __ge__(self, other):
        if self.__comparison_operators_work_as_python_expects:
            return self._compare(other) >= 0
        return self.__binary_expression(">=", other)

    def __binary_expression(self, operator, other, reverse=False):
        other = self._force_formula(other)
        a, b = (self, other) if not reverse else (other, self)
        return Expression(operator, a, b)

    def __add__(self, other):
        return self.__binary_expression("+", other)

    def __radd__(self, other):
        return self.__binary_expression("+", other, reverse=True)

    def __sub__(self, other):
        return self.__binary_expression("-", other)

    def __rsub__(self, other):
        return self.__binary_expression("-", other, reverse=True)

    def __mul__(self, other):
        return self.__binary_expression("*", other)

    def __rmul__(self, other):
        return self.__binary_expression("*", other, reverse=True)

    def __truediv__(self, other):
        return self.__binary_expression("/", other)

    def __rtruediv__(self, other):
        return self.__binary_expression("/", other, reverse=True)

    def __floordiv__(self, other):
        return self.__binary_expression("//", other)

    def __rfloordiv__(self, other):
        return self.__binary_expression("//", other, reverse=True)

    def __mod__(self, other):
        return self.__binary_expression("%", other)

    def __rmod__(self, other):
        return self.__binary_expression("%", other, reverse=True)

    def __pow__(self, other):
        return self.__binary_expression("**", other)

    def __rpow__(self, other):
        return self.__binary_expression("**", other, reverse=True)

    def __lshift__(self, other):
        return self.__binary_expression("<<", other)

    def __rlshift__(self, other):
        return self.__binary_expression("<<", other, reverse=True)

    def __rshift__(self, other):
        return self.__binary_expression(">>", other)

    def __rrshift__(self, other):
        return self.__binary_expression(">>", other, reverse=True)

    def __and__(self, other):
        return self.__binary_expression("&", other)

    def __rand__(self, other):
        return self.__binary_expression("&", other, reverse=True)

    def __or__(self, other):
        return self.__binary_expression("|", other)

    def __ror__(self, other):
        return self.__binary_expression("|", other, reverse=True)

    def __xor__(self, other):
        return self.__binary_expression("^", other)

    def __rxor__(self, other):
        return self.__binary_expression("^", other, reverse=True)

    def __invert__(self):
        return Expression("~", self)

    def __neg__(self):
        return Expression("-", self)

    def __pos__(self):
        return Expression("+", self)

    def __getattr__(self, name):
        name = self._force_formula(name, symbol=True)
        return Expression(".", self, name)

    def __getitem__(self, key):
        key = self._force_formula(key)
        return Expression("[]", self, key)


class ArgsMixin:
    def __init__(self, *args: Formula, min_arity: int = 1):
        if len(args) < min_arity:
            raise ValueError(
                f"{type(self).__name__} must have at least {min_arity} argument{'s' if min_arity > 1 else ''}"
            )
        for arg in args:
            if not isinstance(arg, Formula):
                raise ValueError(f"All arguments must be of type {Formula.__name__}, while {arg} is not")
        self.args = tuple(args)

    @property
    def arity(self):
        return len(self.args)

    def _check_args_equality(self, other: "ArgsMixin"):
        return self.arity == other.arity and all(a._check_equality(b) for a, b in zip(self.args, other.args))

    def _compare_args(self, other: "ArgsMixin"):
        if self.arity != other.arity:
            return -1 if self.arity < other.arity else 1
        for a, b in zip(self.args, other.args):
            cmp = a._compare(b)
            if cmp != 0:
                return cmp
        return 0


class Expression(Formula, ArgsMixin):
    def __init__(self, operator: str, *args: Formula):
        self.operator = str(operator)
        ArgsMixin.__init__(self, *args)

    def _check_equality(self, other):
        return (
            other is not None
            and isinstance(other, Expression)
            and self.operator == other.operator
            and self.arity == other.arity
            and all(a._check_equality(b) for a, b in zip(self.args, other.args))
        )

    def _attributes_to_hash(self):
        return self.operator, *self.args

    def _compare(self, other):
        if not isinstance(other, Expression):
            return -1
        if self.operator != other.operator:
            return -1 if self.operator < other.operator else 1
        return self._compare_args(other)

    def __str__(self):
        if self.arity == 1:
            return f"({self.operator}{self.args[0]})"
        elif self.arity == 2:
            match self.operator:
                case ".":
                    return f"{self.args[0]}.{self.args[1]}"
                case "[]":
                    return f"{self.args[0]}[{self.args[1]}]"
                case _:
                    return f"({self.args[0]} {self.operator} {self.args[1]})"
        else:
            return f"{self.operator}({', '.join(map(str, self.args))})"

    def __repr__(self):
        return f"{type(self).__name__}({self.operator!r}, {', '.join(map(repr, self.args))})"

    def __evaluate(self, operator: str, arity: int, first_arg: object, other_args: list, **kwargs):
        if arity == 2:
            second_arg = other_args[0]
            if operator == ".":
                return getattr(first_arg, second_arg)
            elif operator == "[]":
                args = second_arg
                if isinstance(second_arg, Iterable):
                    args = tuple(
                        [arg.evaluate(**kwargs) if isinstance(arg, Formula) else arg for arg in second_arg]
                    )
                return first_arg[args]  # type: ignore[index]
            elif operator == "+":
                return first_arg + second_arg
            elif operator == "-":
                return first_arg - second_arg
            elif operator == "*":
                return first_arg * second_arg
            elif operator == "/":
                return first_arg / second_arg
            elif operator == "//":
                return first_arg // second_arg
            elif operator == "%":
                return first_arg % second_arg
            elif operator == "**":
                return first_arg**second_arg
            elif operator == "<<":
                return first_arg << second_arg
            elif operator == ">>":
                return first_arg >> second_arg
            elif operator == "&":
                return first_arg & second_arg
            elif operator == "|":
                return first_arg | second_arg
            elif operator == "^":
                return first_arg ^ second_arg
            elif operator == "==":
                return first_arg == second_arg
            elif operator == "!=":
                return first_arg != second_arg
            elif operator == "<":
                return first_arg < second_arg
            elif operator == "<=":
                return first_arg <= second_arg
            elif operator == ">":
                return first_arg > second_arg
            elif operator == ">=":
                return first_arg >= second_arg
        elif arity == 1:
            if operator == "~":
                return ~first_arg  # type: ignore[operator]
            elif operator == "+":
                return +first_arg  # type: ignore[operator]
            elif operator == "-":
                return -first_arg  # type: ignore[operator]
        raise EvaluationError(self, f"Operator {operator} with arity {arity} is not supported")

    def __call__(self, *args):
        arguments = list(self.args[:-1])
        arguments.append(self.args[-1](*args))
        return Expression(self.operator, *arguments)

    def _evaluate(self, **kwargs):
        if self.operator in kwargs:
            return kwargs[self.operator](*self.args, **kwargs)
        if self.arity == 2 and self.operator == "." and isinstance(self.args[1], Predicate):
            receiver = self.args[0].evaluate(**kwargs)
            method = getattr(receiver, self.args[1].functor)
            args = [arg.evaluate(**kwargs) for arg in self.args[1].args]
            return method(*args)
        first = self.args[0].evaluate(**kwargs)
        others = [arg.evaluate(**kwargs) for arg in self.args[1:]]
        return self.__evaluate(self.operator, self.arity, first, others, **kwargs)


class Term(Formula):
    def __init__(self, value: object, symbol: str = None):
        if isinstance(value, Formula):
            raise ValueError("A Term's value cannot be a Formula")
        self.value = value
        self.symbol = symbol
        if symbol is None and hasattr(value, "__name__"):
            self.symbol = value.__name__
        if symbol is not None:
            self.symbol = str(symbol)

    def _check_equality(self, other):
        return (
            other is not None
            and isinstance(other, Term)
            and self.value == other.value
            and self.symbol == other.symbol
        )

    def _attributes_to_hash(self):
        return self.value, self.symbol

    def _compare(self, other):
        if isinstance(other, Expression):
            return 1
        if not isinstance(other, Term):
            return -1
        if self.symbol != other.symbol:
            return -1 if self.symbol < other.symbol else 1
        if self.symbol is not None and self.symbol == other.symbol:
            return 0
        if self.value != other.value:
            return -1 if self.value < other.value else 1
        return 0

    def __str__(self):
        return self.symbol or str(self.value)

    def __repr__(self):
        if self.symbol is None:
            return f"{type(self).__name__}({self.value})"
        return f"{type(self).__name__}({self.value!r}, {self.symbol!r})"

    def _evaluate(self, **kwargs):
        if self.symbol is not None and self.symbol in kwargs:
            return kwargs[self.symbol]
        return self.value

    def __call__(self, *args):
        if self.symbol is None:
            raise ValueError("Term must have a symbol to be used as functors")
        return Predicate(self.symbol, *[self._force_formula(arg) for arg in args])


class Symbol(Formula):
    def __init__(self, name: str):
        self.name = name
        if name is not None:
            self.name = str(name)

    def _check_equality(self, other):
        return other is not None and isinstance(other, Symbol) and self.name == other.name

    def _attributes_to_hash(self):
        return (self.name,)

    def _compare(self, other):
        if isinstance(other, Expression) or isinstance(other, Term):
            return 1
        if not isinstance(other, Symbol):
            return -1
        if self.name != other.name:
            return -1 if self.name < other.name else 1
        return 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"

    def _evaluate(self, **kwargs):
        if self.name in kwargs:
            return kwargs[self.name]
        return self.name

    def __call__(self, *args):
        return Predicate(self.name, *[self._force_formula(arg) for arg in args])


class Predicate(Formula, ArgsMixin):
    def __init__(self, functor: str, *args: Formula):
        self.functor = functor
        ArgsMixin.__init__(self, *args, min_arity=0)

    def _check_equality(self, other):
        return (
            other is not None
            and isinstance(other, Predicate)
            and self.functor == other.functor
            and self._check_args_equality(other)
        )

    def _attributes_to_hash(self):
        return self.functor, *self.args

    def _compare(self, other):
        if isinstance(other, Expression) or isinstance(other, Term) or isinstance(other, Symbol):
            return 1
        if not isinstance(other, Predicate):
            return -1
        if self.functor != other.functor:
            return -1 if self.functor < other.functor else 1
        return self._compare_args(other)

    def __str__(self):
        return f"{self.functor}({', '.join(map(str, self.args))})"

    def __repr__(self):
        result = f"{type(self).__name__}({self.functor!r}"
        if self.arity > 0:
            result += f", {', '.join(map(repr, self.args))}"
        return result + ")"

    def _evaluate(self, **kwargs):
        function = kwargs.get(self.functor)
        if function is None or not callable(function):
            raise EvaluationError(self, f"No viable grounding for functor: {self.functor}", kwargs)
        args = [arg.evaluate(**kwargs) for arg in self.args]
        return function(*args)


class SymbolProvider(Dict[str, object]):
    def __getitem__(self, key):
        return Symbol(key)

    def __contains__(self, key):
        return isinstance(key, str)


def parse(string: str) -> Formula:
    with Formula._comparison_operators_as_builders():
        return eval(string, SymbolProvider())


# let this be the last line of this file
logger.info("skilang loaded")
