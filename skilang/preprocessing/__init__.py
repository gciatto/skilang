import ast


class ChainComparisonTransformer(ast.NodeTransformer):
    def visit_Compare(self, node):
        self.generic_visit(node)

        if len(node.ops) <= 1:
            return node  # Not a chained comparison

        # Build left-associative nested Compare nodes
        left = node.left
        for op, comparator in zip(node.ops, node.comparators):
            new_node = ast.Compare(left=left, ops=[op], comparators=[comparator])
            left = new_node

        return left


def add_left_associative_parentheses_if_chained_comparisons(expr: str) -> str:
    try:
        tree = ast.parse(expr, mode="eval")
        transformed = ChainComparisonTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)
    except SyntaxError:
        return expr


def preprocessing(string: str) -> str:
    """
    Preprocess the input string to add left-associative parentheses to chained comparisons.
    """
    return add_left_associative_parentheses_if_chained_comparisons(string)
