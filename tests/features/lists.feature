Feature: Skilang lists

  Scenario: Simple list usage
    Given the expression "v[0]"
    And "v" defined as "[1, 2, 3]"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the number "1"

  Scenario: Slicing a list
    Given the expression "v[1:3]"
    And "v" defined as "[1, 2, 3, 4, 5]"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the list "[2, 3]"

  Scenario: Multidimensional list
    Given the expression "v[1][1]"
    And "v" defined as "[[1, 2], [3, 4]]"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the number "4"

  Scenario: Multidimensional list with slicing
    Given the expression "v[:][1]"
    And "v" defined as "[[1, 2], [3, 4, 5]]"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the list "[3, 4, 5]"
