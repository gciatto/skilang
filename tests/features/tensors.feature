Feature: Skilang (pytorch) tensors handling

  Scenario: Simple tensor
    Given the expression "t[1]"
    And "t" defined as "tensor([1, 2, 3])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the number "2"

  Scenario: Multidimensional tensor
    Given the expression "t[1, 2]"
    And "t" defined as "tensor([[1, 2, 3], [4, 5, 6]])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the number "6"

  Scenario: Tensor slicing
    Given the expression "t[1:]"
    And "t" defined as "tensor([1, 2, 3, 4, 5])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the tensor "tensor([2, 3, 4, 5])"

  Scenario: Tensor slicing with step
    Given the expression "t[::2]"
    And "t" defined as "tensor([1, 2, 3, 4, 5])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the tensor "tensor([1, 3, 5])"

  Scenario: Multidimensional tensor slicing
    Given the expression "t[:, 2]"
    And "t" defined as "tensor([[1, 2, 3], [4, 5, 6]])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the tensor "tensor([3, 6])"

  Scenario: Operation with tensor
    Given the expression "t[:, i] + t[:, i]"
    And "t" defined as "tensor([[1, 2, 3], [4, 5, 6]])"
    And "i" defined as "2"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the tensor "tensor([6, 12])"

  Scenario: Tensor comparison
    Given the expression "t1[1:] == t2[1:]"
    And "t1" defined as "tensor([1, 2, 4])"
    And "t2" defined as "tensor([1, 2, 5])"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the tensor "tensor([True, False])"

  Scenario: Multidimensional tensor comparison
    Given the expression "t[:, i] == t[:, i]"
    And "t" defined as "tensor([[1, 2, 3], [4, 5, 6]])"
    And "i" defined as "2"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the tensor "tensor([True, True])"

  Scenario: Tridimensional tensor comparison
    Given the expression "t[:, i, 0] == t[:, i, 0]"
    And "t" defined as "tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])"
    And "i" defined as "1"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the tensor "tensor([True, True])"