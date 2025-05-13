from itertools import islice
from typing import List, Dict, Callable

from skilang import Formula, parse


class Rule:
    def __init__(self, name: str, clause: str):
        self.name: str = name
        self.clause: str = clause

    def __str__(self) -> str:
        return f"Rule(name={self.name}, clause={self.clause})"

    def __repr__(self) -> str:
        return f"Rule(name={self.name}, clause={self.clause})"


def get_rules(specification: dict) -> List[Rule]:
    knowledge = specification["knowledge"]
    rules = [Rule(name=rule["rule"], clause=__preprocess_string(rule["clause"])) for rule in knowledge]
    return rules


def __preprocess_string(string: str) -> str:
    return string.replace("[", "[:, ").replace("\n", " ")


def get_knowledge(specification: dict, base_assignments: Dict) -> Dict[str, Callable]:
    knowledge: Dict[str, Callable] = {}

    def create_rule_definition(parsed_rule: Formula, index: int) -> Callable:
        def rule_definition(hand):
            base_assignments["hand"] = hand
            knowledge_so_far: Dict = dict(islice(knowledge.items(), index))
            knowledge_so_far.update(base_assignments)
            return parsed_rule.evaluate(**knowledge_so_far)

        return rule_definition

    rules: List[Rule] = get_rules(specification)
    for index, rule in enumerate(rules):
        parsed_rule: Formula = parse(rule.clause)
        rule_definition: Callable = create_rule_definition(parsed_rule, index)
        knowledge[rule.name] = rule_definition

    return knowledge
