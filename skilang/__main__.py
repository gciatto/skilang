from skilang import *
from pandas import DataFrame


expr = parse("data.x[0] + 1")

print(expr) # (data.x[0] + 1)
print(repr(expr)) # Expression('+', Expression('[]', Expression('.', Symbol('data'), Symbol('x')), Term(0)), Term(1))
print(expr.evaluate(data=DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})))  # 2