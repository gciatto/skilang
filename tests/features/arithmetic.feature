Feature: Skilang arithmetic expressions

  Scenario: Simple expression with all basic operators
    Given the expression "1 * 4 + 2 - 3 / 3"
     When the expression is parsed
     Then the result should be the number "5"

  Scenario: Simple expression with parenthesis
    Given the expression "(2 + 3) * (4 - 1)"
     When the expression is parsed
     Then the result should be the number "15"

  Scenario: Negative result
    Given the expression "(5 - 7) / 2"
     When the expression is parsed
     Then the result should be the number "-1"

  Scenario: Floating point operations
    Given the expression "3.5 + 2.1"
     When the expression is parsed
     Then the result should be the number "5.6"

  Scenario: Expression with exponentiation
    Given the expression "2**2 + 3**3"
     When the expression is parsed
     Then the result should be the number "31"

  Scenario: Expression with multiple levels of nesting
    Given the expression "((1 + (2 * (3 + 4))) - (5 / (6 - 4)))"
     When the expression is parsed
     Then the result should be the number "12.5"

  Scenario: Expression with floor division and modulus
    Given the expression "17 // 3 + 17 % 3"
     When the expression is parsed
     Then the result should be the number "7"
