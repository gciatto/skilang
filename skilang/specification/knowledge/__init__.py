from dataclasses import dataclass
from itertools import islice
from typing import List, Dict, Callable

from skilang import Formula, parse
from skilang.utils import preprocess_batch_tensors


@dataclass
class Rule:
    name: str
    clause: Formula


def get_knowledge(specification: dict) -> List[Rule]:
    knowledge = specification["knowledge"]
    rules = [
        Rule(name=rule["rule"], clause=parse(preprocess_batch_tensors(rule["clause"]))) for rule in knowledge
    ]
    return rules


def get_rules_assignments(knowledge: List[Rule], base_assignments: Dict) -> Dict[str, Callable]:
    rules_assignments: Dict[str, Callable] = {}

    def create_rule_definition(parsed_rule: Formula, index: int) -> Callable:
        def rule_definition(hand):
            # print(hand.size())
            base_assignments["hand"] = hand  # TODO
            knowledge_so_far: Dict = dict(islice(rules_assignments.items(), index))
            knowledge_so_far.update(base_assignments)
            return parsed_rule.evaluate(**knowledge_so_far)

        return rule_definition

    for index, rule in enumerate(knowledge):
        rule_definition: Callable = create_rule_definition(rule.clause, index)
        rules_assignments[rule.name] = rule_definition

    return rules_assignments
