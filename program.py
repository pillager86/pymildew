from sys import stderr

from mildew import Interpreter, LexerError, ParseError, ScriptRuntimeError

def main():
    interpreter = Interpreter()
    while True:
        try:
            line = input("mildew> ").strip()
            if line.strip() == "":
                break
            while len(line) > 0 and line[-1] == '\\':
                line = line[0:-1] + "\n"
                line += input(">>> ")
        except EOFError:
            print("\nEnd of input found. Terminating.")
            break

        #interpreter.evaluate(line)
        try:
            interpreter.evaluate(line)
        except LexerError as lex_error:
            print("\nLexerError: " + str(lex_error), file=stderr)
            print("at " + str(lex_error.position), file=stderr)
        except ParseError as parse_error:
            print("ParseError: " + str(parse_error), file=stderr)
            print("at " + str(parse_error.position), file=stderr)
            print("Token: " + str(parse_error.token), file=stderr)
        except ScriptRuntimeError as sr_error:
            print("ScriptRuntimeError: " + str(sr_error), file=stderr)
            print("Node: " + str(sr_error.node), file=stderr)

if __name__ == "__main__":
    main()

