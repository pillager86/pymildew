from inspect import signature

###############################################################################
# PUBLIC INTERFACE
###############################################################################

class Interpreter:
    '''Holds a context for variables and evaluates scripts'''
    def __init__(self):
        '''Constructs a new Interpreter'''
        self.global_context = ScriptContext(None)
        self.current_context = self.global_context
        # set builtins
        self.global_context.force_set_variable("parseInt", native_parseInt)
        self.global_context.force_set_variable("parseFloat", native_parseFloat)

    def evaluate(self, text):
        '''Evaluates a set of statements'''
        lexer = Lexer(text)
        token = lexer.next_token()
        while(token.type != TT_EOF):
            print(token, end='')
            token = lexer.next_token()
        print()
        parser = Parser(text)
        statement_list = parser.parse()
        program_block = BlockNode(statement_list)
        print(program_block) # temporary, to see node structures
        visit_result = self._visit(program_block)
        return visit_result.value

    def set_global(self, name, value):
        self.global_context.force_set_variable(name, value)

    def _visit(self, tree):
        if isinstance(tree, LiteralNode):
            return VisitResult(tree.value)
        elif isinstance(tree, UnaryOpNode):
            visit_result = self._visit(tree.node)
            value = visit_result.value
            return VisitResult(typesafe_unary_op(tree.op_token, value))
        elif isinstance(tree, BinaryOpNode):
            left_visit_result = self._visit(tree.left_node)
            right_visit_result = self._visit(tree.right_node)
            left_value = left_visit_result.value
            right_value = right_visit_result.value
            result = UNDEFINED

            if tree.op_token.type == TT_ASSIGN:
                if left_visit_result.var_ref is None:
                    raise ScriptRuntimeError(tree, "Cannot assign to lvalue", tree.op_token)
                self.current_context.reassign_variable(tree.left_node.id_token.text, right_value)
                result = right_value
            else:
                # raise ScriptRuntimeError(tree, "Unsupported binary operation", tree.op_token)
                result = typesafe_binary_op(tree.op_token, left_value, right_value)

            # TODO propagate negative and positive infinity through typesafe math operations instead of erroring
            if result is INFINITY or result is NEG_INFINITY:
                raise ScriptRuntimeError(tree, "Division by zero error", tree.op_token)

            return VisitResult(result)
        elif isinstance(tree, VarAccessNode):
            var_ref = self.current_context.access_variable(tree.id_token.text)
            if not self.current_context.var_exists(tree.id_token.text):
                raise ScriptRuntimeError(tree, "Undefined variable", tree.id_token)
            visit_result = VisitResult(var_ref.value)
            visit_result.var_ref = var_ref
            return visit_result
        elif isinstance(tree, ExpressionStatementNode):
            visit_result = self._visit(tree.node)
            print("Expression statement result: " + str(visit_result.value)) # temporary
            # temporary: expression statements have their effect through function calls and should not return 
            #            a value. Only expressions themselves should return a value
            return visit_result
        elif isinstance(tree, BlockNode):
            self.current_context = ScriptContext(self.current_context)
            visit_result = VisitResult(UNDEFINED)
            for each_node in tree.statement_nodes:
                visit_result = self._visit(each_node)
                if visit_result.break_flag or visit_result.return_flag:
                    break
            self.current_context = self.current_context.parent
            return visit_result
        elif isinstance(tree, VarDeclarationNode):
            if tree.kw_spec_token.text == "let":
                declfunc = self.current_context.declare_variable
            else: # TODO implement const eventually
                declfunc = self.global_context.declare_variable
            for i in range(len(tree.var_tokens)):
                value = UNDEFINED
                visit_result = VisitResult(UNDEFINED)
                if tree.expression_nodes[i] is not None:
                    visit_result = self._visit(tree.expression_nodes[i])
                success = declfunc(tree.var_tokens[i].text, visit_result.value)
                if not success:
                    raise ScriptRuntimeError(tree, "Cannot redeclare variable " + tree.var_tokens[i].text, tree.var_tokens[i])
            return VisitResult(UNDEFINED) # this can't return anything meaningful
        elif isinstance(tree, IfStatementNode):
            condition_result = self._visit(tree.condition_node)
            visit_result = VisitResult(UNDEFINED)
            if condition_result.value:
                visit_result = self._visit(tree.if_statement)
            else:
                if tree.else_statement is not None:
                    visit_result = self._visit(tree.else_statement)
            return visit_result
        elif isinstance(tree, WhileStatementNode):
            visit_result_condition = self._visit(tree.condition_node)
            loop_result = VisitResult(UNDEFINED)
            while visit_result_condition.value:
                # TODO check for return value set and exit if so
                loop_result = self._visit(tree.loop_statement)
                if loop_result.break_flag:
                    loop_result.break_flag = False
                    break
                if loop_result.return_flag:
                    break
                visit_result_condition = self._visit(tree.condition_node)
            return loop_result
        elif isinstance(tree, DoWhileStatementNode):
            loop_result = self._visit(tree.loop_statement) # do at least once
            condition_result = self._visit(tree.condition_node)
            while condition_result.value:
                if loop_result.break_flag:
                    loop_result.break_flag = False
                    break
                if loop_result.return_flag:
                    break
                loop_result = self._visit(tree.loop_statement)
                condition_result = self._visit(tree.condition_node)
            return loop_result
        elif isinstance(tree, ForStatementNode):
            self.current_context = ScriptContext(self.current_context)
            self._visit(tree.init_statement)
            condition_result = VisitResult(True)
            body_result = VisitResult(UNDEFINED)
            if tree.condition_node is not None:
                condition_result = self._visit(tree.condition_node)
            while condition_result.value:
                body_result = self._visit(tree.loop_statement)
                if body_result.break_flag:
                    body_result.break_flag = False
                    break
                if body_result.return_flag:
                    break
                self._visit(tree.increment_node)
                if tree.condition_node is None:
                    condition_result.value = True
                else:
                    condition_result = self._visit(tree.condition_node)
            self.current_context = self.current_context.parent
            return body_result
        elif isinstance(tree, BreakStatementNode):
            visit_result = VisitResult(UNDEFINED)
            visit_result.break_flag = True
            return visit_result
        elif isinstance(tree, ReturnStatementNode):
            visit_result = VisitResult(UNDEFINED)
            if tree.expression_node is not None:
                visit_result = self._visit(tree.expression_node)
            visit_result.return_flag = True
            visit_result.return_value = visit_result.value
            return visit_result
        elif isinstance(tree, FunctionCallNode):
            fn_to_call = self._visit(tree.fn_expression)
            if fn_to_call.var_ref is None:
                raise ScriptRuntimeError(tree, "Cannot call function " + str(fn_to_call.value))
            args_to_pass = []
            for arg in tree.arg_expressions:
                arg_visit = self._visit(arg)
                args_to_pass.append(arg_visit.value)
            # is this a native Python function?
            if callable(fn_to_call.value):
                sig = signature(fn_to_call.value)
                if(len(args_to_pass) != len(sig.parameters)):
                    raise ScriptRuntimeError(tree, "Wrong number of args for native function. " \
                        + f"Expected {len(sig.parameters)} got {len(args_to_pass)}")
                return VisitResult(fn_to_call.value(*args_to_pass))
            else:
                raise ScriptRuntimeError(tree, "Cannot call a non function")
        elif tree is None:
            return VisitResult(UNDEFINED) # nothing to do (empty statement)
        else:
            raise ScriptRuntimeError(tree, "Cannot visit unknown node type " + str(type(tree)))
        
        return VisitResult(UNDEFINED)

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
    def __init__(self, node, msg, token=None):
        super().__init__(msg)
        self.node = node
        self.token = token

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
TT_BIT_AND          = "BIT_AND"     # &
TT_BIT_OR           = "BIT_OR"      # |
TT_BIT_XOR          = "BIT_XOR"     # ^
TT_BIT_NOT          = "BIT_NOT"     # ~
TT_BIT_LSHIFT       = "BIT_LSHIFT"  # <<
TT_BIT_RSHIFT       = "BIT_RSHIFT"  # >>
TT_LPAREN           = "LPAREN"      # (
TT_RPAREN           = "RPAREN"      # )
TT_LBRACE           = "LBRACE"      # {
TT_RBRACE           = "RBRACE"      # }
TT_SEMICOLON        = "SEMICOLON"   # ;
TT_COMMA            = "COMMA"       # ,

###############################################################################
# KEYWORDS
###############################################################################
KW_TRUE             = "true"
KW_FALSE            = "false"
KW_UNDEFINED        = "undefined"
KW_VAR              = "var"
KW_LET              = "let"
KW_IF               = "if"
KW_ELSE             = "else"
KW_WHILE            = "while"
KW_DO               = "do"
KW_FOR              = "for"
KW_BREAK            = "break"
KW_RETURN           = "return"
KEYWORDS = [KW_TRUE, KW_FALSE, KW_UNDEFINED, KW_VAR, KW_LET, KW_IF, KW_ELSE, 
            KW_WHILE, KW_DO, KW_FOR, KW_BREAK, KW_RETURN]

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
        # TODO support // and /* ... */ comments
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
            elif self._peek_char() == '>':
                self._advance_char()
                token = Token(TT_BIT_RSHIFT, self.position)
            else:
                token = Token(TT_GT, self.position)
        elif self._current_char() == '<':
            if self._peek_char() == '=':
                self._advance_char()
                token = Token(TT_LE, self.position)
            elif self._peek_char() == '<':
                self._advance_char()
                token = Token(TT_BIT_LSHIFT, self.position)
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
                token = Token(TT_BIT_AND, self.position)
        elif self._current_char() == '|':
            if self._peek_char() == '|':
                self._advance_char()
                token = Token(TT_OR, self.position)
            else:
                token = Token(TT_BIT_OR, self.position)
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
        elif self._current_char() == '^':
            token = Token(TT_BIT_XOR, self.position)
        elif self._current_char() == '~':
            token = Token(TT_BIT_NOT, self.position)
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
    if token.type in [TT_BIT_NOT, TT_NOT, TT_PLUS, TT_DASH]:
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
    elif token.type in [TT_BIT_RSHIFT, TT_BIT_LSHIFT]:
        return 13
    elif token.type in [TT_LT, TT_LE, TT_GT, TT_GE]:
        return 12
    elif token.type in [TT_EQUALS, TT_NEQUALS]:
        return 11
    elif token.type == TT_BIT_AND:
        return 10
    elif token.type == TT_BIT_XOR:
        return 9
    elif token.type == TT_BIT_OR:
        return 8
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

    # build a statement list and return it (also the "program" grammar rule)
    def parse(self):
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
                    if self._current_token.type != TT_IDENTIFIER:
                        raise ParseError(self.lexer.position, self._current_token, "Trailing commas are not allowed in variable declarations")

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
        # is it an if-statement?
        elif self._current_token.is_keyword(KW_IF):
            self._advance()
            if self._current_token.type != TT_LPAREN:
                raise ParseError(self.lexer.position, self._current_token, "Expected '(' after if keyword")
            self._advance()
            condition_node = self._expr()
            if self._current_token.type != TT_RPAREN:
                raise ParseError(self.lexer.position, self._current_token, "Expected ')' after if condition")
            self._advance()
            if_node = self._statement()
            else_node = None
            if self._current_token.is_keyword(KW_ELSE):
                self._advance()
                else_node = self._statement()
            statement_node = IfStatementNode(condition_node, if_node, else_node)
        # is it a while statement?
        elif self._current_token.is_keyword(KW_WHILE):
            self._advance()
            if self._current_token.type != TT_LPAREN:
                raise ParseError(self.lexer.position, self._current_token, "Expected '(' after while keyword")
            self._advance()
            condition_node = self._expr()
            if self._current_token.type != TT_RPAREN:
                raise ParseError(self.lexer.position, self._current_token, "Expected ')' after while condition")
            self._advance()
            loop_statement = self._statement()
            statement_node = WhileStatementNode(condition_node, loop_statement)
        # is it a do-while statement?
        elif self._current_token.is_keyword(KW_DO):
            self._advance()
            loop_statement = self._statement()
            if not self._current_token.is_keyword(KW_WHILE):
                raise ParseError(self.lexer.position, self._current_token, "Expected while keyword after do block")
            self._advance()
            if self._current_token.type != TT_LPAREN:
                raise ParseError(self.lexer.position, self._current_token, "Expected '(' after while")
            self._advance()
            condition_node = self._expr()
            if self._current_token.type != TT_RPAREN:
                raise ParseError(self.lexer.position, self._current_token, "Expected ')' after do while loop condition")
            self._advance()
            if self._current_token.type != TT_SEMICOLON:
                raise ParseError(self.lexer.position, self._current_token, "Expected ';' after do-while statement")
            self._advance()
            statement_node = DoWhileStatementNode(loop_statement, condition_node)
        # is it a for statement?
        elif self._current_token.is_keyword(KW_FOR):
            self._advance()
            if self._current_token.type != TT_LPAREN:
                raise ParseError(self.lexer.position, self._current_token, "Expected '(' after for keyword")
            self._advance()
            init_statement = self._statement()
            if self._current_token.type == TT_SEMICOLON:
                condition_node = None
                self._advance()
            else:
                condition_node = self._expr()
                if self._current_token.type != TT_SEMICOLON:
                    raise ParseError(self.lexer.position, self._current_token, "Expected ';' after condition expression in for loop")
                else:
                    self._advance()
            if self._current_token.type == TT_RPAREN:
                increment_node = None
            else:
                increment_node = self._expr()
            if self._current_token.type != TT_RPAREN:
                raise ParseError(self.lexer.position, self._current_token, "Expected ')' after for loop conditions")
            self._advance()
            loop_statement = self._statement()
            statement_node = ForStatementNode(init_statement, condition_node, increment_node, loop_statement)
        # a break statement?
        elif self._current_token.is_keyword(KW_BREAK):
            statement_node = BreakStatementNode(self._current_token)
            self._advance()
            if self._current_token.type != TT_SEMICOLON:
                raise ParseError(self.lexer.position, self._current_token, "Expected ';' after break")
            self._advance()
        # a return statement?
        elif self._current_token.is_keyword(KW_RETURN):
            self._advance()
            expression_node = None
            if self._current_token.type != TT_SEMICOLON:
                expression_node = self._expr()
            if self._current_token.type != TT_SEMICOLON:
                raise ParseError(self.lexer.position, self._current_token, "Expected ';' after return statement")
            statement_node = ReturnStatementNode(expression_node)
        # could be an empty statement or program
        elif self._current_token.type == TT_SEMICOLON or self._current_token.type == TT_EOF:
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

        un_op_prec = unary_op_priority(self._current_token)
        if un_op_prec != 0 and un_op_prec >= parent_precedence:
            token = self._current_token
            self._advance()
            operand = self._expr(un_op_prec)
            left = UnaryOpNode(token, operand)
        else:
            left = self._primary_expr()

        # cheap hack to enforce right-assoc on POW and ASSIGN. TODO implement this with data
        if self._current_token.type == TT_POW or self._current_token.type == TT_ASSIGN:
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
        # TODO check for postfix increment/decrement operators on identifiers
        #       will also have to account for member access (.) situations
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
        # we have a primary-expr now see if there are parentheses turning this into a function call
        if self._current_token.type == TT_LPAREN:
            self._advance()
            arg_expressions = []
            while self._current_token.type != TT_RPAREN and self._current_token.type != TT_EOF:
                arg_expressions.append(self._expr())
                if self._current_token.type == TT_COMMA:
                    self._advance()
                elif self._current_token.type == TT_RPAREN:
                    self._advance()
                    break
            left = FunctionCallNode(left, arg_expressions)
        # TODO this is where we will handle the dot operator as well as array access
        #      also where to handle postfix and prefix operator possibly
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

class ExpressionStatementNode:
    def __init__(self, node):
        self.node = node
    def __repr__(self):
        return f"Expression statement: {self.node}"

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

class IfStatementNode:
    def __init__(self, condition_node, if_statement, else_statement=None):
        self.condition_node = condition_node
        self.if_statement = if_statement
        self.else_statement = else_statement
    def __repr__(self):
        rep = "if(" + str(self.condition_node) + ") " + str(self.if_statement)
        if self.else_statement is not None:
            rep += " else " + str(self.else_statement)
        return rep

class WhileStatementNode:
    def __init__(self, condition_node, loop_statement):
        self.condition_node = condition_node
        self.loop_statement = loop_statement
    def __repr__(self):
        return f"while({self.condition_node}) {self.loop_statement}"

class DoWhileStatementNode:
    def __init__(self, loop_statement, condition_node):
        self.loop_statement = loop_statement
        self.condition_node = condition_node
    def __repr__(self):
        return f"do {self.loop_statement} while({self.condition_node})"

class ForStatementNode:
    def __init__(self, init_statement, condition_node, increment_node, loop_statement):
        self.init_statement = init_statement
        self.condition_node = condition_node
        self.increment_node = increment_node
        self.loop_statement = loop_statement
    def __repr__(self):
        return f"for({self.init_statement};{self.condition_node};{self.increment_node}) {self.loop_statement}"

class BreakStatementNode:
    def __init__(self, break_token):
        self.break_token = break_token
    def __repr__(self):
        return "Break statement"

class ReturnStatementNode:
    def __init__(self, expression_node = None):
        self.expression_node = expression_node
    def __repr__(self):
        return f"return {self.expression_node}"

class FunctionCallNode:
    def __init__(self, fn_expression, arg_expressions):
        self.fn_expression = fn_expression # most likely should be a var access
        self.arg_expressions = arg_expressions
    def __repr__(self):
        rep = f"Function call: {self.fn_expression}("
        for arg in range(len(self.arg_expressions)):
            rep += str(self.arg_expressions[arg])
            if arg < len(self.arg_expressions) - 1:
                rep += ", "
        rep += ")"
        return rep

class FunctionDeclarationNode:
    def __init__(self, id_token, arg_name_tokens, statement_nodes):
        self.id_token = id_token
        self.arg_name_tokens = arg_name_tokens
        self.statement_nodes = statement_nodes
    def __repr__(self):
        rep = f"Function declaration: {self.id_token.text}("
        for arg in range(len(self.arg_name_tokens)):
            rep += str(self.arg_name_tokens[arg].text)
            if arg < len(self.arg_name_tokens) - 1:
                rep += ", "
        rep += ") {"
        for statement_node in self.statement_nodes:
            rep += f"\t{statement_node}"
        rep += "}"
        return rep



###############################################################################
# FUNCTION TYPE
###############################################################################
class ScriptFunction:
    def __init__(self, arg_names, statement_nodes, name="<anonymous function>"):
        self.arg_names = arg_names
        self.statement_nodes = statement_nodes
        self.name = name
    def __repr__(self):
        rep = f"function {self.name}{tuple(self.arg_names)}" + "{"
        for statement_node in self.statement_nodes:
            rep += f"\t{statement_node}"
        rep += "}"
        return rep

###############################################################################
# RETURN VALUE FOR INTERPRETER VISIT FUNCTION
###############################################################################

class VisitResult:
    def __init__(self, value):
        self.value = value
        self.var_ref = None
        self.return_value = UNDEFINED
        self.return_flag = False
        self.break_flag = False
    def __repr__(self):
        return f"value={self.value}, var_ref={self.var_ref}, return_value={self.return_value}, " \
                + f"return_flag={self.return_flag}, break_flag={self.break_flag}"

###############################################################################
# TYPESAFE OPERATIONS
###############################################################################

# TODO handle infinity as a valid value

def arg_is_numerical(arg):
    return type(arg) == bool or type(arg) == int or type(arg) == float or isinstance(arg, Infinity)

def typesafe_binary_op(op_token, left, right):
    if left is UNDEFINED or right is UNDEFINED:
        return UNDEFINED
    # equality can be tested on anything
    if op_token.type == TT_EQUALS:
        return left == right
    elif op_token.type == TT_NEQUALS:
        return left != right
    # one string argument equals string type coercion
    if type(left) == str or type(right) == str:
        left = str(left)
        right = str(right)
    # for logical boolean operators it should be safe to use on any type
    if op_token.type == TT_AND:
        return left and right 
    elif op_token.type == TT_OR:
        return left or right
    # for + they can only be numeric or str
    if op_token.type == TT_PLUS:
        if (arg_is_numerical(left) or type(left) == str) and (arg_is_numerical(right) or type(right) == str):
            return left + right
        else:
            return UNDEFINED
    # for -,*,/,%,**, &, ^, | they must be a number
    if op_token.type in [TT_DASH, TT_STAR, TT_FSLASH, TT_PERCENT, TT_POW, \
                         TT_BIT_AND, TT_BIT_XOR, TT_BIT_OR, TT_BIT_LSHIFT, TT_BIT_RSHIFT]:
        if not arg_is_numerical(left) or not arg_is_numerical(right):
            return UNDEFINED
    if op_token.type == TT_DASH:
        return left - right
    elif op_token.type == TT_STAR:
        return left * right
    elif op_token.type == TT_FSLASH:
        if right == 0:
            if left < 0:
                return NEG_INFINITY
            elif left > 0:
                return INFINITY
            else:
                return UNDEFINED
        return left / right
    elif op_token.type == TT_PERCENT:
        return left % right
    elif op_token.type == TT_POW:
        return left ** right
    elif op_token.type == TT_BIT_AND:
        return left & right
    elif op_token.type == TT_BIT_XOR:
        return left ^ right
    elif op_token.type == TT_BIT_OR:
        return left | right
    elif op_token.type == TT_BIT_LSHIFT:
        return left << right
    elif op_token.type == TT_BIT_RSHIFT:
        return left >> right
    # for comparisons they must both be a numeric type OR the exact same type
    if not(arg_is_numerical(left) and arg_is_numerical(right)):
        if type(left) != type(right):
            return UNDEFINED
    if op_token.type == TT_GE:
        return left > right
    elif op_token.type == TT_GT:
        return left >= right
    elif op_token.type == TT_LE:
        return left < right
    elif op_token.type == TT_LT:
        return left <= right
    print("Warning, unknown binary operator " + op_token.type)
    return UNDEFINED

def typesafe_unary_op(op_token, operand):
    if operand is UNDEFINED:
        return UNDEFINED
    # not can be used on anything
    if op_token.type == TT_NOT:
        return not operand
    # plus and minus and bit_not can ONLY be used on numeric
    if not arg_is_numerical(operand):
        return UNDEFINED
    if op_token.type == TT_PLUS:
        return operand
    elif op_token.type == TT_DASH:
        return operand * -1
    elif op_token.type == TT_BIT_NOT:
        return ~operand
    print("Warning, unknown unary operator " + op_token.type)
    return UNDEFINED


# singleton for default values of uninitialized variables. propagates through typesafe_ operations
class Undefined:
    def __bool__(self):
        return False
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

INFINITY = Infinity()
NEG_INFINITY = Infinity(True)

###############################################################################
# RUNTIME CONTEXT
###############################################################################

# Since Python doesn't have pointers we wrap variable values in a reference like this
class VarReference:
    def __init__(self, value):
        self.value = value

class ScriptContext:
    # TODO a const table that will be checked on each var declare or assign attempt
    def __init__(self, parent=None):
        self.parent = parent
        self.variables = {}

    def access_variable(self, name):
        context = self
        while context is not None and not name in context.variables:
            context = context.parent
        if context is None:
            return None
        else:
            return context.variables[name]

    def reassign_variable(self, name, value):
        # to be used in reassignment not declaration
        context = self
        while context is not None and not name in context.variables:
            context = context.parent
        if context is None:
            return None
        else:
            if value is UNDEFINED:
                del context.variables[name]
                return None
            else:
                context.variables[name].value = value
                return context.variables[name]

    # this only declares a local variable in this context that may shadow globals
    def declare_variable(self, name, value=UNDEFINED):
        if not name in self.variables:
            self.variables[name] = VarReference(value)
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

    # to be used by the host environment to set globals
    def force_set_variable(self, name, value):
        self.variables[name] = VarReference(value)

###############################################################################
# ESSENTIAL BUILTIN FUNCTIONS
###############################################################################

def native_parseInt(string):
    try:
        return int(string)
    except ValueError:
        return UNDEFINED

def native_parseFloat(string):
    try:
        return float(string)
    except ValueError:
        return UNDEFINED
