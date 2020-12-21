import sys

from mildew import Interpreter, LexerError, ParseError, ScriptRuntimeError

def eval_with_error_checking(interpreter, text, file_name = "<stdin>"):
    result = None
    try:
        result = interpreter.evaluate(text)
    except LexerError as lex_error:
        print("\nLexerError in " + file_name + ": " + str(lex_error), file=sys.stderr)
        print("at " + str(lex_error.position), file=sys.stderr)
    except ParseError as parse_error:
        print("ParseError in " + file_name + ": " + str(parse_error), file=sys.stderr)
        print("at " + str(parse_error.position), file=sys.stderr)
        print("Token: " + str(parse_error.token), file=sys.stderr)
    except ScriptRuntimeError as sr_error:
        print("ScriptRuntimeError in " + file_name + ": " + str(sr_error), file=sys.stderr)
        print("Node: " + str(sr_error.node), file=sys.stderr)
        if sr_error.token is not None:
            print("Token: " + str(sr_error.token) + " at " + str(sr_error.token.position), file=sys.stderr)
    return result

# TODO we need something better than input() that allows arrow keys to be used
def main(args):
    interpreter = Interpreter()
    if len(args) > 1:
        file_to_read = args[1]
        input_file = open(file_to_read, "r")
        code = input_file.read()
        result = eval_with_error_checking(interpreter, code, file_to_read)
        print("The result is " + str(result))
    else:
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
            result = eval_with_error_checking(interpreter, line)
            print("The result is " + str(result))


if __name__ == "__main__":
    main(sys.argv)

