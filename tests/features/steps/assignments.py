from typing import Callable

import pandas as pd
import torch
from behave import use_step_matcher, given, when

from skilang import Formula

use_step_matcher("parse")


@given('"{name}" defined as "{definition}"')
def step_assignment(context, name: str, definition: str):
    """
    :type context: behave.runner.Context
    :param name: the name of the symbol
    :param definition: the assignment definition
    :return:
    """
    eval_context = {"DataFrame": pd.DataFrame, "tensor": torch.tensor}
    obj: Callable = eval(definition, eval_context)

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
        raise ValueError("Assignments not provided.")
    parsed: Formula = context.parsed_exp
    context.result = parsed.evaluate(**context.assignments)
