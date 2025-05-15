from dataclasses import dataclass
from itertools import islice
from typing import List, Dict, Callable

from skilang import Formula, parse


@dataclass
class Rule:
    name: str
    clause: Formula


def __preprocess_string(string: str) -> str:
    return string.replace("[", "[:, ").replace("\n", " ")


def get_knowledge(specification: dict) -> List[Rule]:
    knowledge = specification["knowledge"]
    rules = [Rule(name=rule["rule"], clause=parse(__preprocess_string(rule["clause"]))) for rule in knowledge]
    return rules


def get_rules_assignments(specification: dict, base_assignments: Dict) -> Dict[str, Callable]:
    knowledge: Dict[str, Callable] = {}

    def create_rule_definition(parsed_rule: Formula, index: int) -> Callable:
        def rule_definition(hand):
            base_assignments["hand"] = hand
            knowledge_so_far: Dict = dict(islice(knowledge.items(), index))
            knowledge_so_far.update(base_assignments)
            return parsed_rule.evaluate(**knowledge_so_far)

        return rule_definition

    rules: List[Rule] = get_knowledge(specification)
    for index, rule in enumerate(rules):
        rule_definition: Callable = create_rule_definition(rule.clause, index)
        knowledge[rule.name] = rule_definition

    return knowledge
