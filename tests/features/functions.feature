Feature: Skilang functions

  Scenario: Function definition
    Given the expression "f(x)"
    When the expression is parsed
    Then the result should be a function

  Scenario: Simple function call
    Given the expression "f(2)"
    And "f" defined as "lambda x: x + 1"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the number "3"

  Scenario: Function call with multiple arguments
    Given the expression "f(2, 1, 3)"
    And "f" defined as "lambda x, y, z: x + y + z"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the number "6"

  Scenario: Function call with symbolic arguments
    Given the expression "f(x, y, 2)"
    And "f" defined as "lambda x, y, z: x if y else z"
    And "x" defined as "1"
    And "y" defined as "True"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the number "1"

  Scenario: Function call with strings
    Given the expression "g('Hello', a)"
    And "g" defined as "lambda x, y: x + ' ' + y + '!'"
    And "a" defined as "'World'"
    When the expression is parsed
    And the expression is evaluated
    Then the result should be the string "Hello World!"
