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
    context.parsed_exp = context.result = parsed


@when("the expression is evaluated")
def step_expression_evaluated(context):
    """
    :type context: behave.runner.Context
    """
    if "parsed_exp" not in context:
        raise ValueError("Parsed expression not provided.")
    if "assignments" not in context:
        raise ValueError("Assignments not provided.")
    parsed: Formula = context.parsed_exp
    context.result = parsed.evaluate(**context.assignments)
