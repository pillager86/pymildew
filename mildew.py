###############################################################################
# PUBLIC INTERFACE
###############################################################################

class Interpreter:
    '''Holds a context for variables and evaluates expressions'''
    def __init__(self):
        '''Constructs a new Interpreter'''
        self.global_context = ScriptContext(None)
        self.current_context = self.global_context

    def evaluate(self, text):
        '''Evaluates a set of statements'''
        # create a context just for this evaluate
        self.current_context = ScriptContext(self.current_context)
        lexer = Lexer(text)
        token = lexer.next_token()
        while(token.type != TT_EOF):
            print(token, end='')
            token = lexer.next_token()
        print()
        parser = Parser(text)
        statement_list = parser.parse()
        for statement_node in statement_list:
            print(statement_node)
            self._visit(statement_node)
        # pop local context
        self.current_context = self.current_context.parent
        return statement_list

    def _visit(self, tree):
        if isinstance(tree, LiteralNode):
            return tree.value
        elif isinstance(tree, UnaryOpNode):
            # TODO: refactor, collect node value up front and check to make sure unary op makes sense for type
            if tree.op_token.type == TT_NOT:
                return typesafe_logical_not(self._visit(tree.node))
            elif tree.op_token.type == TT_DASH:
                return typesafe_unary_minus(self._visit(tree.node))
            elif tree.op_token.type == TT_PLUS:
                return typesafe_unary_plus(self._visit(tree.node))
            else:
                raise ScriptRuntimeError(tree, "Unknown unary operator")
        elif isinstance(tree, BinaryOpNode):
            left_value = self._visit(tree.left_node)
            right_value = self._visit(tree.right_node)
            result = UNDEFINED

            if tree.op_token.type == TT_OR:
                result = typesafe_logical_or(left_value, right_value)
            elif tree.op_token.type == TT_AND:
                result = typesafe_logical_and(left_value, right_value)
    
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
                result = typesafe_neq(left_value, right_value)

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

            if result is INFINITY:
                raise ScriptRuntimeError(tree, "Division by zero error")

            return result
        elif isinstance(tree, VarAccessNode):
            result = self.current_context.access_variable(tree.id_token.text)
            if not self.current_context.var_exists(tree.id_token.text):
                raise ScriptRuntimeError(tree, "Undefined variable")
            return result
        elif isinstance(tree, VarAssignmentNode):
            # TODO handle decrement and increment -= += assignments as well
            if not self.current_context.var_exists(tree.var_token.text):
                raise ScriptRuntimeError(tree, "Cannot assign to undeclared variable")
            else:
                return self.current_context.reassign_variable(tree.var_token.text, self._visit(tree.right_node))
        elif isinstance(tree, ExpressionStatementNode):
            result = self._visit(tree.node)
            print("Expression statement result: " + str(result)) # temporary
        elif isinstance(tree, BlockNode):
            self.current_context = ScriptContext(self.current_context)
            for each_node in tree.statement_nodes:
                result = self._visit(each_node)
            self.current_context = self.current_context.parent
        elif isinstance(tree, VarDeclarationNode):
            # value = UNDEFINED
            # if tree.expression_node is not None:
            #     value = self._visit(tree.expression_node)
            # if tree.kw_spec_token.text == "let":
            #     self.current_context.declare_variable(tree.var_token.text, value)
            # else:
            #     self.global_context.declare_variable(tree.var_token.text, value)
            if tree.kw_spec_token.text == "let":
                declfunc = self.current_context.declare_variable
            else:
                declfunc = self.global_context.declare_variable
            for i in range(len(tree.var_tokens)):
                value = UNDEFINED
                if tree.expression_nodes[i] is not None:
                    value = self._visit(tree.expression_nodes[i])
                declfunc(tree.var_tokens[i].text, value)
            result = UNDEFINED # this type of expression cannot evaluate to anything
        elif tree is None:
            return UNDEFINED # nothing to do
        else:
            raise ScriptRuntimeError(tree, "Cannot visit unknown node type " + str(type(tree)))
        
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
TT_LBRACE           = "LBRACE"      # {
TT_RBRACE           = "RBRACE"      # }
TT_SEMICOLON        = "SEMICOLON"   # ;
TT_COMMA            = "COMMA"       # ,

###############################################################################
# KEYWORDS
###############################################################################
KW_TRUE = "true"
KW_FALSE = "false"
KW_UNDEFINED = "undefined"
KW_VAR = "var"
KW_LET = "let"
KEYWORDS = [ KW_TRUE, KW_FALSE, KW_UNDEFINED, KW_VAR, KW_LET ]

###############################################################################
# LEXER CLASSES
###############################################################################

ESCAPE_CHARS = {
    'b': '\b', 'f': '\f',
    'n': '\n', 'r': '\r', 't': '\t', 
    'v': '\v', '0': '\0', "'": '\'',
    '"': '\"', '\\': '\\'
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
    def is_keyword(self, kw_name):
        if self.type != TT_KEYWORD:
            return False
        if self.text == kw_name:
            return True
        else:
            return False

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
            result = ""
            while self._current_char() != closing_quote:
                if self._current_char() == '\0':
                    raise LexerError(self.position, "Missing closing quotation mark for string literal")
                elif self._current_char() == '\\':
                    self._advance_char()
                    if self._current_char() in ESCAPE_CHARS:
                        result += ESCAPE_CHARS[self._current_char()]
                    else:
                        raise LexerError(self.position, "Unknown escape char `" + self._current_char + "`")
                else:
                    result += self._current_char()
                self._advance_char()
            token = Token(TT_STRING, startpos, result)
            
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
        # STATEMENT TOKENS
        elif self._current_char() == '{':
            token = Token(TT_LBRACE, self.position)
        elif self._current_char() == '}':
            token = Token(TT_RBRACE, self.position)
        elif self._current_char() == ';':
            token = Token(TT_SEMICOLON, self.position)
        elif self._current_char() == ',':
            token = Token(TT_COMMA, self.position)
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

    # build a tree and return it (also the "program" grammar rule)
    def parse(self):
        # left = self._expr()
        # if self._current_token.type != TT_EOF: # no trailing garbage allowed
        #    raise ParseError(self._current_token.position, self._current_token, "Unexpected token after expression")
        # return left
        stmts = []
        while self._current_token.type != TT_EOF:
            stmts.append(self._statement())
        return stmts

    # grammar rules

    def _statement(self):
        statement_node = None
        # is it a var declaration
        if self._current_token.is_keyword(KW_VAR) or self._current_token.is_keyword(KW_LET):
            var_spec_token = self._current_token
            self._advance()
            id_tokens = []
            assign_expressions = []
            if self._current_token.type != TT_IDENTIFIER:
                raise ParseError(self.lexer.position, self._current_token, "Expected identifier")

            while self._current_token.type == TT_IDENTIFIER:
                id_tokens.append(self._current_token)
                self._advance()
                # there is an assignment
                if self._current_token.type == TT_ASSIGN:
                    self._advance()
                    assign_expressions.append(self._expr())
                else:
                    assign_expressions.append(None)
                if self._current_token.type == TT_COMMA:
                    self._advance()

            statement_node = VarDeclarationNode(var_spec_token, id_tokens, assign_expressions)

            # ensure semicolon
            if self._current_token.type != TT_SEMICOLON:
                raise ParseError(self.lexer.position, self._current_token, "Missing semicolon")

            self._advance()
        # is it a {} block
        elif self._current_token.type == TT_LBRACE:
            self._advance()
            stmts = []
            while self._current_token.type != TT_RBRACE and self._current_token.type != TT_EOF:
                stmts.append(self._statement())
            if self._current_token.type != TT_RBRACE:
                raise ParseError(self.lexer.position, self._current_token, "Expected '}'")
            self._advance()                
            statement_node = BlockNode(stmts)
        # could be an empty statement
        elif self._current_token.type == TT_SEMICOLON:
            self._advance()
            statement_node = ExpressionStatementNode(None)
        # else it has to be an expression followed by a semicolon
        else:
            statement_node = ExpressionStatementNode(self._expr())
            if self._current_token.type != TT_SEMICOLON:
                raise ParseError(self.lexer.position, self._current_token, "Missing ';'")
            self._advance()

        return statement_node

    def _expr(self, parent_precedence = 0):
        left = None

        # check for assignment by peeking next token
        # TODO make this a separate grammar rule once statements are introduced
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
            elif self._current_token.text == KW_UNDEFINED:
                left = LiteralNode(UNDEFINED)
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

class ExpressionStatementNode:
    def __init__(self, node):
        self.node = node
    def __repr__(self):
        return f"(Statement: {self.node})"

class BlockNode:
    def __init__(self, statement_nodes):
        self.statement_nodes = statement_nodes
    def __repr__(self):
        rep = "{"
        for st_node in self.statement_nodes:
            rep += str(st_node) + "\n"
        rep += "}"
        return rep

class VarDeclarationNode:
    def __init__(self, kw_spec_token, var_tokens, expression_nodes):
        self.kw_spec_token = kw_spec_token
        self.var_tokens = var_tokens
        self.expression_nodes = expression_nodes
        # if this is used properly the assert should never fail
        assert (len(self.var_tokens) == len(self.expression_nodes))
    def __repr__(self):
        rep = self.kw_spec_token.text + " "
        for i in range(len(self.var_tokens)):
            rep += self.var_tokens[i].text
            if self.expression_nodes[i] is not None:
                rep += "=" + str(self.expression_nodes[i])
            if i < len(self.var_tokens) - 1:
                rep += ", "
        return rep

###############################################################################
# TYPESAFE OPERATIONS
###############################################################################

# TODO refactor to avoid code duplication

def typesafe_logical_and(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    else:
        return left and right

def typesafe_logical_or(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    else:
        return left or right

def typesafe_logical_not(operand):
    if operand is UNDEFINED:
        return UNDEFINED
    else:
        return not operand

def typesafe_unary_plus(operand):
    if operand is UNDEFINED or type(operand) == str:
        return UNDEFINED
    else:
        return operand

def typesafe_unary_minus(operand):
    if operand is UNDEFINED or type(operand) == str:
        return UNDEFINED
    else:
        return operand * -1

def typesafe_add(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return str(left) + str(right)
    else:
        return left + right

def typesafe_sub(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return UNDEFINED
    else:
        return left - right

def typesafe_mul(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return UNDEFINED
    else:
        return left * right

def typesafe_div(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return UNDEFINED
    elif right == 0:
        return INFINITY
    else:
        return left / right

def typesafe_mod(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return UNDEFINED
    else:
        return left % right

def typesafe_pow(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return UNDEFINED
    else:
        return left ** right

def typesafe_eq(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    else:
        return left == right

def typesafe_neq(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    else:
        return left != right

def typesafe_lt(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return str(left) < str(right)
    else:
        return left < right

def typesafe_le(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return str(left) <= str(right)
    else:
        return left <= right

def typesafe_gt(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return str(left) > str(right)
    else:
        return left > right

def typesafe_ge(left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    elif type(left) == str or type(right) == str:
        return str(left) >= str(right)
    else:
        return left >= right

# singleton for default values of uninitialized variables. propagates through typesafe_ operations
class Undefined:
    def __repr__(self):
        return "undefined"

UNDEFINED = Undefined()

# TODO propagate positive or negative Infinity through math ops instead of erroring out
class Infinity:
    def __init__(self, is_negative=False):
        self.is_negative = is_negative
    def __repr__(self):
        if not self.is_negative:
            return "Infinity"
        else:
            return "-Infinity"

# TODO rather than use this singleton, check for isinstance(Infinity) when propagating Infinity value from division by zero
INFINITY = Infinity()

###############################################################################
# RUNTIME CONTEXT
###############################################################################

class ScriptContext:
    def __init__(self, parent=None):
        self.parent = parent
        self.variables = {}

    def access_variable(self, name):
        context = self
        while context is not None and not name in context.variables:
            context = context.parent
        if context is None:
            return UNDEFINED
        else:
            return context.variables[name]

    def reassign_variable(self, name, value):
        # to be used in reassignment not declaration
        context = self
        while context is not None and not name in context.variables:
            context = context.parent
        if context is None:
            return UNDEFINED
        else:
            if value is UNDEFINED:
                del context.variables[name]
            else:
                context.variables[name] = value
            return value

    def declare_variable(self, name, value=UNDEFINED):
        if not name in self.variables:
            self.variables[name] = value
            return True
        else:
            return False

    def var_exists(self, name):
        context = self
        while context is not None and not name in context.variables:
            context = context.parent
        if context is None:
            return False
        else:
            return True
