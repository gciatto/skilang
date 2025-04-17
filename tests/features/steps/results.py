from behave import use_step_matcher, then


use_step_matcher("parse")


@then('the result should be the number "{expected}"')
def step_number_result(context, expected: int):
    """
    :type context: behave.runner.Context
    :param expected: the expected result
    """
    if "parsed_exp" not in context:
        raise ValueError("Parsed expression not provided.")
    assert float(expected) == context.parsed_exp, f"Expected {float(expected)}, but got {context.parsed_exp}"
