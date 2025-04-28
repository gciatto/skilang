from torch import Tensor


def parse_hand(hand: Tensor) -> str:
    result = ""
    for i in range(5):
        suit: int = int(hand[i * 2].item())
        rank: int = int(hand[i * 2 + 1].item())
        result += parse_card(suit, rank) + "\n"
    return result


def parse_card(suit: int, rank: int) -> str:
    return f"{get_rank(rank)} of {get_suit(suit)}"


def get_rank(rank: int) -> str:
    if rank == 1:
        return "Ace"
    elif rank == 11:
        return "Jack"
    elif rank == 12:
        return "Queen"
    elif rank == 13:
        return "King"
    else:
        return str(rank)


def get_suit(index: int) -> str:
    if index == 1:
        return "Hearts"
    elif index == 2:
        return "Spades"
    elif index == 3:
        return "Diamonds"
    elif index == 4:
        return "Clubs"
    raise ValueError(f"Index {index} not recognized")


def get_class_name(index: int) -> str:
    if index == 0:
        return "Nothing in hand"
    elif index == 1:
        return "One pair"
    elif index == 2:
        return "Two pairs"
    elif index == 3:
        return "Three of a kind"
    elif index == 4:
        return "Straight"
    elif index == 5:
        return "Flush"
    elif index == 6:
        return "Full house"
    elif index == 7:
        return "Four of a kind"
    elif index == 8:
        return "Straight flush"
    elif index == 9:
        return "Royal flush"
    raise ValueError(f"Index {index} not recognized")
