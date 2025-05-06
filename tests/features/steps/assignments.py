from typing import Callable

from behave import use_step_matcher, given, when

from skilang import Formula, skilang_builtins_dict

use_step_matcher("parse")


@given('"{name}" defined as "{definition}"')
def step_assignment(context, name: str, definition: str):
    """
    :type context: behave.runner.Context
    :param name: the name of the symbol
    :param definition: the assignment definition
    :return:
    """
    obj: Callable = eval(definition, skilang_builtins_dict())

    if "assignments" not in context:
        context.assignments = {name: obj}
    else:
        context.assignments[name] = obj


@when("the expression is evaluated")
def step_expression_evaluated(context):
    """
    :type context: behave.runner.Context
    """
    if "parsed_exp" not in context:
        raise ValueError("Parsed expression not provided.")
    if "assignments" not in context:
        context.assignments = {}
    parsed: Formula = context.parsed_exp
    context.result = parsed.evaluate(**context.assignments)
