import torch
from behave import use_step_matcher, then

from skilang import Predicate

use_step_matcher("parse")


@then('the result should be the number "{expected}"')
def step_number_result(context, expected: str):
    """
    :type context: behave.runner.Context
    :param expected: the expected result
    """
    if "result" not in context:
        raise ValueError("Actual result not provided.")
    assert float(expected) == context.result, f"Expected {float(expected)}, but got {context.result}"


@then('the result should be the string "{expected}"')
def step_string_result(context, expected: str):
    """
    :type context: behave.runner.Context
    :param expected: the expected result
    """
    if "result" not in context:
        raise ValueError("Actual result not provided.")
    assert expected == context.result, f"Expected {expected}, but got {context.result}"


@then("the result should be a function")
def step_function_result(context):
    """
    :type context: behave.runner.Context
    """
    if "result" not in context:
        raise ValueError("Actual result not provided.")
    assert isinstance(context.result, Predicate), f"Expected a function, but got {type(context.result)}"


@then('the result should be the list "{expected}"')
def step_list_result(context, expected: str):
    """
    :type context: behave.runner.Context
    """
    if "result" not in context:
        raise ValueError("Actual result not provided.")
    assert context.result == eval(expected), f"Expected {expected}, but got {context.result}"


@then('the result should be the tensor "{expected}"')
def step_tensor_result(context, expected: str):
    """
    :type context: behave.runner.Context
    """
    if "result" not in context:
        raise ValueError("Actual result not provided.")
    eval_context = {"tensor": torch.tensor}
    assert torch.equal(context.result, eval(expected, eval_context)), (
        f"Expected {expected}, but got {context.result}"
    )
