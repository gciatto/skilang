import logging
import typing


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('skilang')


class EvaluationError(Exception):
    def __init__(self, formula):
        super().__init__(f"Error evaluating formula: {formula}")


class Formula:
    def evaluate(self, **kwargs):
        try:
            return self._evaluate(**kwargs)
        except Exception as e:
            raise EvaluationError(self) from e
    
    def _evaluate(self, **kwargs):
        raise NotImplementedError
    
    def __force_formula(self, value, symbol: bool = False) -> 'Formula':
        if isinstance(value, Formula):
            return value
        if symbol:
            return Symbol(value)
        return Term(value)
    
    def __add__(self, other):
        other = self.__force_formula(other)
        return Expression("+", self, other)

    def __radd__(self, other):
        other = self.__force_formula(other)
        return Expression("+", other, self)
    
    def __sub__(self, other):
        other = self.__force_formula(other)
        return Expression("-", self, other)

    def __rsub__(self, other):
        other = self.__force_formula(other)
        return Expression("-", other, self)
    
    def __mul__(self, other):
        other = self.__force_formula(other)
        return Expression("*", self, other)

    def __rmul__(self, other):
        other = self.__force_formula(other)
        return Expression("*", other, self)
    
    def __truediv__(self, other):
        other = self.__force_formula(other)
        return Expression("/", self, other)

    def __rtruediv__(self, other):
        other = self.__force_formula(other)
        return Expression("/", other, self)
    
    def __getattr__(self, name):
        name = self.__force_formula(name, symbol=True)
        return Expression(".", self, name)
    
    def __getitem__(self, key):
        key = self.__force_formula(key)
        return Expression("[]", self, key)


class Expression(Formula):
    __operators_map ={
        "+": "__add__",
        "-": "__sub__",
        "*": "__mul__",
        "/": "__truediv__",
        ".": "__getattr__",
        "[]": "__getitem__",
    }

    def __init__(self, operator: str, *args: Formula):
        self.operator = str(operator)
        if not args:
            raise ValueError("Expression must have at least one argument")
        if any(not isinstance(arg, Formula) for arg in args):
            raise ValueError(f"All arguments must be of type {type(Formula).__name__}")
        self.args = tuple(args)

    @property
    def arity(self):
        return len(self.args)
    
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

    def _evaluate(self, **kwargs):
        first = self.args[0].evaluate(**kwargs)
        others = [arg.evaluate(**kwargs) for arg in self.args[1:]]
        method = self.__operators_map[self.operator] if self.operator in self.__operators_map else self.operator
        return getattr(first, method)(*others)
        

class Term(Formula):
    def __init__(self, value: object, symbol: str = None):
        self.value = value
        self.symbol = symbol
        if symbol is not None:
            self.symbol = str(symbol)

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


class Symbol(Formula):
    def __init__(self, name: str):
        self.name = name
        if name is not None:
            self.name = str(name)

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"
    
    def _evaluate(self, **kwargs):
        if self.name in kwargs:
            return kwargs[self.name]
        return self.name
    

class SymbolProvider(typing.Dict[str, object]):
    def __getitem__(self, key):
        return Symbol(key)
    
    def __contains__(self, key):
        return isinstance(key, str)
    

def parse(string: str) -> Formula:
    return eval(string, SymbolProvider())


# let this be the last line of this file
logger.info("skilang loaded")
