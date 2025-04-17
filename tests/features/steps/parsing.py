from behave import use_step_matcher, given, when

from skilang import parse, Formula

use_step_matcher("parse")


@given('the expression "{exp}"')
def step_expression_input(context, exp: str):
    """
    :type context: behave.runner.Context
    :param exp: the expression to be parsed
    """
    context.exp = exp


@when("the expression is parsed")
def step_expression_parsed(context):
    """
    :type context: behave.runner.Context
    """
    if "exp" not in context:
        raise ValueError("Expression not provided.")
    parsed: Formula = parse(context.exp)
    context.result = context.parsed_exp = parsed
