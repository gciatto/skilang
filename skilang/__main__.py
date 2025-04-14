from skilang import parse


while True:
    try:
        string = input("> ")
    except (EOFError, KeyboardInterrupt):
        break
    expression = parse(string)
    print("\t", str(expression))
    print("\t", repr(expression))
