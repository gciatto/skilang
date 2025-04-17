Feature: Skilang bitwise operations

  Scenario: AND operation
    Given the expression "6 & 4"
    When the expression is parsed
    Then the result should be the number "4"

  Scenario: OR operation
    Given the expression "2 | 1"
    When the expression is parsed
    Then the result should be the number "3"

  Scenario: XOR operation
    Given the expression "5 ^ 3"
    When the expression is parsed
    Then the result should be the number "6"

  Scenario: NOT operation
    Given the expression "~0"
    When the expression is parsed
    Then the result should be the number "-1"

  Scenario: Left shift operation
    Given the expression "3 << 2"
    When the expression is parsed
    Then the result should be the number "12"

  Scenario: Right shift operation
    Given the expression "8 >> 1"
    When the expression is parsed
    Then the result should be the number "4"