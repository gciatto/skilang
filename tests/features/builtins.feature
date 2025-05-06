Feature: Skilang builtins

  Scenario: Contains with lists and tuples
    Given the expression "contains([1, 2, 3, 4], 2) and contains((1, 2, 3, 4), 1)"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the boolean "True"

  Scenario: Contains with tensor
    Given the expression "contains(tensor([1, 2, 3]), 2)"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the boolean "True"

  Scenario: Contains with multidimensional tensor
    Given the expression "contains(t, 2)"
    And "t" defined as "tensor([[1, 2, 3], [4, 5, 6]])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the tensor "tensor([True, False])"

  Scenario: Concat two lists
    Given the expression "concat([1, 2, 3], [4, 5, 6])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the list "[1, 2, 3, 4, 5, 6]"

  Scenario: Concat two tensors
    Given the expression "concat(t1, t2)"
    And "t1" defined as "tensor([1, 2, 3])"
    And "t2" defined as "tensor([4, 5, 6])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the tensor "tensor([1, 2, 3, 4, 5, 6])"

  Scenario: Variance of a tensor
    Given the expression "variance(t)"
    And "t" defined as "tensor([1, 2, 3])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the number "1"

  Scenario: Variance of a list
    Given the expression "variance([1, 2, 3, 4, 5])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the number "2.5"

