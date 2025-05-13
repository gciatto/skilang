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


# tris_rule = "((hand[:, rank1] == hand[:, rank2]) & (hand[:, rank1] == hand[:, rank3])) | ((hand[:, rank1] == hand[:, rank2]) & (hand[:, rank1] == hand[:, rank4])) | ((hand[:, rank1] == hand[:, rank2]) & (hand[:, rank1] == hand[:, rank5])) | ((hand[:, rank1] == hand[:, rank3]) & (hand[:, rank1] == hand[:, rank4])) | ((hand[:, rank1] == hand[:, rank3]) & (hand[:, rank1] == hand[:, rank5])) | ((hand[:, rank1] == hand[:, rank4]) & (hand[:, rank1] == hand[:, rank5])) | ((hand[:, rank2] == hand[:, rank3]) & (hand[:, rank2] == hand[:, rank4])) | ((hand[:, rank2] == hand[:, rank3]) & (hand[:, rank2] == hand[:, rank5])) | ((hand[:, rank2] == hand[:, rank4]) & (hand[:, rank2] == hand[:, rank5])) | ((hand[:, rank3] == hand[:, rank4]) & (hand[:, rank3] == hand[:, rank5]))"
#
# two_pair_rule = (
#     "((hand[:, rank1] == hand[:, rank2]) & ((hand[:, rank3] == hand[:, rank4]) | (hand[:, rank3] == hand[:, rank5]) | (hand[:, rank4] == hand[:, rank5]))) | "
#     "((hand[:, rank1] == hand[:, rank3]) & ((hand[:, rank2] == hand[:, rank4]) | (hand[:, rank2] == hand[:, rank5]) | (hand[:, rank4] == hand[:, rank5]))) | "
#     "((hand[:, rank1] == hand[:, rank4]) & ((hand[:, rank2] == hand[:, rank3]) | (hand[:, rank2] == hand[:, rank5]) | (hand[:, rank3] == hand[:, rank5]))) | "
#     "((hand[:, rank1] == hand[:, rank5]) & ((hand[:, rank2] == hand[:, rank3]) | (hand[:, rank2] == hand[:, rank4]) | (hand[:, rank3] == hand[:, rank4])))"
# )
#
#
# t = torch.tensor([[3.0, 8.0, 1.0, 8.0, 1.0, 7.0, 2.0, 6.0, 1.0, 8.0]], device="mps:0")
# # t = torch.tensor([[1., 9., 1., 4., 1., 3., 1., 3., 1., 7.],
# #                   [1., 3., 1., 4., 1., 3., 1., 3., 1., 8.]], device='mps:0')
# assignments: dict = {"rank1": 1, "rank2": 3, "rank3": 5, "rank4": 7, "rank5": 9, "hand": t}
#
# formula: Formula = parse(tris_rule)
# print(formula.__repr__())
# respected_rule: Tensor = formula.evaluate(**assignments)
# print(respected_rule)
# print(respected_rule.sum())
# print(respected_rule.argmax())
