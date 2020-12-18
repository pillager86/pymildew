###############################################################################
# PUBLIC INTERFACE
###############################################################################

class Interpreter:
    '''Holds a context for variables and evaluates expressions'''
    def __init__(self):
        '''Constructs a new Interpreter with a global context to store all variables for now'''
        self.global_context = Context(None)

    def evaluate(self, text):
        '''Evaluates a single expression (for now)'''
        lexer = Lexer(text)
        token = lexer.next_token()
        while(token.type != TT_EOF):
            print(token, end='')
            token = lexer.next_token()
        print()
        parser = Parser(text)
        tree = parser.parse()
        print(tree)
        return self._visit(tree)

    def _visit(self, tree):
        if isinstance(tree, LiteralNode):
            return tree.value
        elif isinstance(tree, UnaryOpNode):
            # TODO: refactor, collect node value up front and check to make sure unary op makes sense for type
            if tree.op_token.type == TT_NOT:
                return not self._visit(tree.node)
            elif tree.op_token.type == TT_DASH:
                return self._visit(tree.node) * -1
            elif tree.op_token.type == TT_PLUS:
                return self._visit(tree.node)
            else:
                raise ScriptRuntimeError(tree, "Unknown unary operator")
        elif isinstance(tree, BinaryOpNode):
            left_value = self._visit(tree.left_node)
            right_value = self._visit(tree.right_node)
            result = UNDEFINED

            if tree.op_token.type == TT_OR:
                result = left_value or right_value
            elif tree.op_token.type == TT_AND:
                result = left_value and right_value
    
            elif tree.op_token.type == TT_LT:
                result = typesafe_lt(left_value, right_value)
            elif tree.op_token.type == TT_LE:
                result = typesafe_le(left_value, right_value)
            elif tree.op_token.type == TT_GT:
                result = typesafe_gt(left_value, right_value)
            elif tree.op_token.type == TT_GE:
                result = typesafe_ge(left_value, right_value)
            elif tree.op_token.type == TT_EQUALS:
                result = typesafe_eq(left_value, right_value)
            elif tree.op_token.type == TT_NEQUALS:
                result = not typesafe_eq(left_value, right_value)

            elif tree.op_token.type == TT_PLUS:
                result = typesafe_add(left_value, right_value)
            elif tree.op_token.type == TT_DASH:
                result = typesafe_sub(left_value, right_value)
            elif tree.op_token.type == TT_STAR:
                result = typesafe_mul(left_value, right_value)
            elif tree.op_token.type == TT_FSLASH:
                result = typesafe_div(left_value, right_value)
            elif tree.op_token.type == TT_PERCENT:
                result = typesafe_mod(left_value, right_value)
            elif tree.op_token.type == TT_POW:
                result = typesafe_pow(left_value, right_value)
            else:
                raise ScriptRuntimeError(tree, "Unsupported binary operation")

            if result is UNDEFINED: # TODO propagate undefined value through operations
                raise ScriptRuntimeError(tree, "Type mismatch in binary operation")

            if result is INFINITY:
                raise ScriptRuntimeError(tree, "Division by zero error")

            return result
        elif isinstance(tree, VarAccessNode):
            result = self.global_context.get_variable(tree.id_token.text)
            if result is UNDEFINED:
                raise ScriptRuntimeError(tree, "Undefined variable accessed")
            return result
        elif isinstance(tree, VarAssignmentNode):
            # TODO handle decrement and increment -= += assignments as well
            return self.global_context.set_variable(tree.var_token.text, self._visit(tree.right_node))
        else:
            raise ScriptRuntimeError(tree, "Cannot visit unknown node type")
        
        return None

###############################################################################
# ERRORS
###############################################################################

class LexerError(Exception):
    def __init__(self, position, msg):
        super().__init__(msg)
        self.position = position

class ParseError(Exception):
    def __init__(self, position, token, msg):
        super().__init__(msg)
        self.position = position
        self.token = token

class ScriptRuntimeError(Exception):
    def __init__(self, node, msg):
        super().__init__(msg)
        self.node = node

###############################################################################
# TOKEN TYPE ENUM
###############################################################################
TT_EOF              = "EOF"
TT_KEYWORD          = "KEYWORD"
TT_INTEGER          = "INTEGER"     
TT_DOUBLE           = "DOUBLE"       
TT_STRING           = "STRING"
TT_KEYWORD          = "KEYWORD"
TT_IDENTIFIER       = "IDENTIFIER"
TT_NOT              = "NOT"         # !
TT_AND              = "AND"         # &&
TT_OR               = "OR"          # ||
TT_GT               = "GT"          # >
TT_GE               = "GE"          # >=
TT_LT               = "LT"          # <
TT_LE               = "LE"          # <=
TT_EQUALS           = "EQUALS"      # ==
TT_NEQUALS          = "NEQUALS"     # !=
TT_ASSIGN           = "ASSIGN"      # =
TT_PLUS             = "PLUS"        # +
TT_DASH             = "DASH"        # -
TT_STAR             = "STAR"        # *
TT_FSLASH           = "FSLASH"      # /
TT_PERCENT          = "PERCENT"     # %
TT_POW              = "POW"         # **
TT_LPAREN           = "LPAREN"      # (
TT_RPAREN           = "RPAREN"      # )

###############################################################################
# KEYWORDS
###############################################################################
KW_TRUE = "true"
KW_FALSE = "false"
KEYWORDS = [ KW_TRUE, KW_FALSE ]

###############################################################################
# LEXER CLASSES
###############################################################################

ESCAPE_CHARS = {
    'n': '\n', 't': '\t', 'r': '\r' # TODO add complete set
}

class Position:
    def __init__(self, line, column):
        self.line = line
        self.column = column

    def advance(self, char):
        if char == '\0':
            return
        elif char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1

    def copy(self):
        return Position(self.line, self.column)

    def __repr__(self):
        return "line: " + str(self.line) + " column: " + str(self.column)

class Token:
    def __init__(self, token_type, position, text=None):
        self.type = token_type
        self.position = position.copy()
        self.text = text
    def __repr__(self):
        result = "[" + self.type
        if self.text is not None and len(self.text) > 0:
            result += ":" + self.text 
        result += "]"
        return result

def starts_keyword_or_id(char):
    return char.isalpha() or char == '_' or char == '$'

def continues_keyword_or_id(char):
    return char.isalnum() or char == '_'  or char == '$'

class Lexer:
    def __init__(self, text):
        self._text = text
        self._char_counter = 0
        self.position = Position(1, 1)

    def _current_char(self):
        if self._char_counter < len(self._text):
            return self._text[self._char_counter]
        else:
            return '\0'

    def _advance_char(self):
        self._char_counter += 1
        self.position.advance(self._current_char())

    def _peek_char(self):
        if self._char_counter + 1 < len(self._text):
            return self._text[self._char_counter + 1]
        else:
            return '\0'

    def next_token(self, consume=True):
        # in case we are only peeking
        old_char_counter = self._char_counter
        old_position = self.position.copy()
        # return value
        token = None
        # handle white space
        while self._current_char().isspace():
            self._advance_char()
        # check for integer or double token
        if self._current_char().isnumeric():
            start = self._char_counter
            startpos = self.position.copy()
            dot_counter = 0
            e_counter = 0
            while self._peek_char().isnumeric() or self._peek_char() == '.' or self._peek_char().lower() == 'e':
                self._advance_char()
                if self._current_char() == '.':
                    dot_counter += 1
                    if dot_counter > 1:
                        raise LexerError(startpos, "Too many decimals in number literal")
                elif self._current_char() == 'e':
                    e_counter += 1
                    if e_counter > 1:
                        raise LexerError(startpos, "Number literal may only have one exponent specifier")
                    # is there an optional '-' or '+'
                    if self._peek_char() in ['+', '-']:
                        # safe to ignore, Python's parser will handle it
                        self._advance_char()
                    # but it MUST be followed by a number
                    if not self._peek_char().isnumeric():
                        raise LexerError(startpos, "Number must follow exponent in number literal")
            text = self._text[start:self._char_counter+1]
            if dot_counter == 0:
                token = Token(TT_INTEGER, startpos, text)
            else:
                token = Token(TT_DOUBLE, startpos, text)
        # check for string literal
        elif self._current_char() == "'" or self._current_char() == '"':
            closing_quote = self._current_char()
            startpos = self.position.copy()
            self._advance_char()
            start = self._char_counter
            while self._current_char() != closing_quote and self._current_char() != '\0':
                # TODO handle '\\' + escape chars
                if self._current_char() == '\0':
                    raise LexerError(startpos, "Missing closing quotation mark for string literal")
                self._advance_char()
            text = self._text[start:self._char_counter]
            token = Token(TT_STRING, startpos, text)
            
        # LOGIC OPERATORS
        elif self._current_char() == '>':
            if self._peek_char() == '=':
                self._advance_char()
                token = Token(TT_GE, self.position)
            else:
                token = Token(TT_GT, self.position)
        elif self._current_char() == '<':
            if self._peek_char() == '=':
                self._advance_char()
                token = Token(TT_LE, self.position)
            else:
                token = Token(TT_LT, self.position)
        elif self._current_char() == '=':
            if self._peek_char() == '=':
                self._advance_char()
                token = Token(TT_EQUALS, self.position)
            else:
                token = Token(TT_ASSIGN, self.position)
        elif self._current_char() == '!':
            if self._peek_char() == '=':
                self._advance_char()
                token = Token(TT_NEQUALS, self.position)
            else:
                token = Token(TT_NOT, self.position)
        elif self._current_char() == '&':
            if self._peek_char() == '&':
                self._advance_char()
                token = Token(TT_AND, self.position)
            else:
                raise LexerError(self.position, "Bitwise and & not supported yet")
        elif self._current_char() == '|':
            if self._peek_char() == '|':
                self._advance_char()
                token = Token(TT_OR, self.position)
            else:
                raise LexerError(self.position, "Bitwise or | not supported yet")
        # MATH OPERATORS
        elif self._current_char() == '+':
            token = Token(TT_PLUS, self.position)
        elif self._current_char() == '-':
            token = Token(TT_DASH, self.position)
        elif self._current_char() == '*':
            if self._peek_char() == '*':
                self._advance_char()
                token = Token(TT_POW, self.position)
            else:
                token = Token(TT_STAR, self.position)
        elif self._current_char() == '/':
            token = Token(TT_FSLASH, self.position)
        elif self._current_char() == '%':
            token = Token(TT_PERCENT, self.position)
        # PARENTHESES
        elif self._current_char() == '(':
            token = Token(TT_LPAREN, self.position)
        elif self._current_char() == ')':
            token = Token(TT_RPAREN, self.position)
        # EOF
        elif self._current_char() == '\0':
            token = Token(TT_EOF, self.position)
        # keyword or identifier
        elif starts_keyword_or_id(self._current_char()):
            start = self._char_counter
            startpos = self.position.copy()
            while continues_keyword_or_id(self._peek_char()):
                self._advance_char()
            text = self._text[start:self._char_counter+1]
            if text in KEYWORDS:
                token = Token(TT_KEYWORD, startpos, text)
            else:
                token = Token(TT_IDENTIFIER, startpos, text)
        else:
            raise LexerError(self.position, "Unknown character " + self._current_char())
        self._advance_char()
        if not consume:
            self._char_counter = old_char_counter
            self.position = old_position
        return token

###############################################################################
# PARSER
###############################################################################

def unary_op_priority(token):
    # see grammar.txt for explanation of magic constants
    if token.type in [TT_NOT, TT_PLUS, TT_DASH]:
        return 17
    else:
        return 0

def binary_op_priority(token):
    # see grammar.txt for explanation of magic constants
    if token.type == TT_POW:
        return 16
    elif token.type in [TT_STAR, TT_FSLASH, TT_PERCENT]:
        return 15
    elif token.type in [TT_PLUS, TT_DASH]:
        return 14
    elif token.type in [TT_LT, TT_LE, TT_GT, TT_GE]:
        return 12
    elif token.type in [TT_EQUALS, TT_NEQUALS]:
        return 11
    elif token.type == TT_AND:
        return 7
    elif token.type == TT_OR:
        return 6
    elif token.type == TT_ASSIGN:
        return 3
    else:
        return 0

class Parser:
    def __init__(self, text):
        self.lexer = Lexer(text)
        self._advance()

    # build a tree and return it
    def parse(self):
        # TODO parse semicolon separated statements instead of one expression
        left = self._expr()
        if self._current_token.type != TT_EOF: # no trailing garbage allowed
            raise ParseError(self._current_token.position, self._current_token, "Unexpected token after expression")
        return left

    # grammar rules

    def _expr(self, parent_precedence = 0):
        left = None

        # check for assignment by peeking next token
        peek = self.lexer.next_token(False)
        if self._current_token.type == TT_IDENTIFIER and peek.type == TT_ASSIGN:
            var_token = self._current_token
            self._advance()
            assign_token = self._current_token
            self._advance()
            # right = self._expr(parent_precedence)
            right = self._expr(binary_op_priority(assign_token))
            left = VarAssignmentNode(assign_token, var_token, right)
            return left

        un_op_prec = unary_op_priority(self._current_token)
        if un_op_prec != 0 and un_op_prec >= parent_precedence:
            token = self._current_token
            self._advance()
            operand = self._expr(un_op_prec)
            left = UnaryOpNode(token, operand)
        else:
            left = self._primary_expr()
        
        # cheap hack to make sure no attempt to assign to an lvalue
        if self._current_token.type == TT_ASSIGN:
            raise ParseError(self._current_token.position, self._current_token, "Cannot assign to an lvalue")

        # cheap hack to enforce right-assoc on POW. TODO implement this with data
        if self._current_token.type == TT_POW:
            prec = binary_op_priority(self._current_token)
            token = self._current_token
            self._advance()
            right = self._expr(prec)
            left = BinaryOpNode(token, left, right)

        # handle normal binary operations with priority and left-associativity
        while True:
            prec = binary_op_priority(self._current_token)
            if prec == 0 or prec <= parent_precedence:
                break
            token = self._current_token
            self._advance()
            right = self._expr(prec)
            left = BinaryOpNode(token, left, right)

        return left

    def _primary_expr(self):
        left = None
        if self._current_token.type == TT_LPAREN:
            self._advance()
            left = self._expr()
            if self._current_token.type != TT_RPAREN:
                raise ParseError(self.lexer.position, self._current_token, "Missing ')'")
            else:
                self._advance()
        elif self._current_token.type == TT_DOUBLE:
            left = LiteralNode(float(self._current_token.text))
            self._advance()
        elif self._current_token.type == TT_INTEGER:
            left = LiteralNode(int(self._current_token.text))
            self._advance()
        elif self._current_token.type == TT_KEYWORD:
            if self._current_token.text == KW_TRUE:
                left = LiteralNode(True)
            elif self._current_token.text == KW_FALSE:
                left = LiteralNode(False)
            else:
                raise ParseError(self._current_token.position, self._current_token, "Unexpected keyword")
            self._advance()
        elif self._current_token.type == TT_STRING:
            left = LiteralNode(self._current_token.text)
            self._advance()
        elif self._current_token.type == TT_IDENTIFIER:
            left = VarAccessNode(self._current_token)
            self._advance()
        else:
            raise ParseError(self._current_token.position, self._current_token, "Unexpected token")
        return left

    # helper functions
    def _advance(self):
        self._current_token = self.lexer.next_token()

        

###############################################################################
# NODES
###############################################################################

class BinaryOpNode:
    def __init__(self, op_token, left_node, right_node):
        self.op_token = op_token
        self.left_node = left_node
        self.right_node = right_node
    def __repr__(self):
        return f"({self.left_node} {self.op_token.type} {self.right_node})"

class UnaryOpNode:
    def __init__(self, op_token, node):
        self.op_token = op_token
        self.node = node
    def __repr__(self):
        return f"({self.op_token.type} {self.node})"

class LiteralNode:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return str(self.value)

class VarAccessNode:
    def __init__(self, id_token):
        self.id_token = id_token
    def __repr__(self):
        return f"var '{self.id_token.text}'"

class VarAssignmentNode:
    def __init__(self, op_token, var_token, right_node):
        self.op_token = op_token
        self.var_token = var_token
        self.right_node = right_node
    def __repr__(self):
        return f"({self.var_token} {self.op_token} {self.right_node})"

###############################################################################
# TYPESAFE OPERATIONS
###############################################################################

# TODO propagate undefined value through operations

def typesafe_add(left, right):
    if type(left) == str or type(right) == str:
        return str(left) + str(right)
    else:
        return left + right

def typesafe_sub(left, right):
    if type(left) == str or type(right) == str:
        return UNDEFINED
    else:
        return left - right

def typesafe_mul(left, right):
    if type(left) == str or type(right) == str:
        return UNDEFINED
    else:
        return left * right

def typesafe_div(left, right):
    if type(left) == str or type(right) == str:
        return UNDEFINED
    elif right == 0:
        return INFINITY
    else:
        return left / right

def typesafe_mod(left, right):
    if type(left) == str or type(right) == str:
        return UNDEFINED
    else:
        return left % right

def typesafe_pow(left, right):
    if type(left) == str or type(right) == str:
        return UNDEFINED
    else:
        return left ** right

def typesafe_eq(left, right):
    return left == right

def typesafe_lt(left, right):
    if type(left) == str or type(right) == str:
        return str(left) < str(right)
    else:
        return left < right

def typesafe_le(left, right):
    if type(left) == str or type(right) == str:
        return str(left) <= str(right)
    else:
        return left <= right

def typesafe_gt(left, right):
    if type(left) == str or type(right) == str:
        return str(left) > str(right)
    else:
        return left > right

def typesafe_ge(left, right):
    if type(left) == str or type(right) == str:
        return str(left) >= str(right)
    else:
        return left >= right

class Undefined:
    def __repr__(self):
        return "undefined"

UNDEFINED = Undefined()

class Infinity:
    def __repr__(self):
        return "infinity"

INFINITY = Infinity()

###############################################################################
# RUNTIME CONTEXT
###############################################################################

class Context: # TODO possible rename to ScriptContext for clarity
    def __init__(self, parent=None):
        self.parent = parent
        self.variables = {}

    def get_variable(self, name):
        # for now just look in current context
        if name in self.variables:
            return self.variables[name]
        else:
            return UNDEFINED

    def set_variable(self, name, value):
        # for now just set in current context
        self.variables[name] = value
        return self.variables[name]
