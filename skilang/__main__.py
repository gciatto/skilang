from skilang import parse

if __name__ == "__main__":
    while True:
        try:
            string = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        expression = parse(string)
        print("\t", str(expression))
        print("\t", f"{repr(expression)}, {type(expression)}")
