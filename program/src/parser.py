from scanner import *
from tree import *

# Important convention: Unary functions have their argument in 'right' and None in 'left'

_desc = None  # Current descriptor
_ident = None  # Current identifier
_scanner = None
_companion = None  # Companion object for that every node of the parsed tree knows
def parse(input):
    # These three get returned at the end
    dag = None    # Stores expression as a binary tree (Until CSE, then it's a binary DAG)
    variable_ranks = {}   # Stores declared variables and their ranks
    arg_name = None   # Stores the derivation argument variable name

    global _desc
    global _ident
    global _scanner
    _scanner = Scanner(input)
    _get_sym()

    global _companion
    _companion = TreeCompanion()

    variable_ranks = _declaration(variable_ranks)
    dag = _expressionpart()
    arg_name = _argument()
    return dag, arg_name, variable_ranks

def _get_sym():
    global _desc
    global _ident
    global scanner
    (_desc, _ident) = _scanner.get_sym()

def _fits(symbol):
    global _desc
    return _desc == symbol

def _error(expected):
    global _ident
    raise Exception(f'Expected {expected} but found \'{_ident}\'')

def _declaration(variable_ranks):
    if _fits(TOKEN_ID.DECLARE):
        _get_sym()
        variable_ranks = _tensordeclaration(variable_ranks)
        while not _fits(TOKEN_ID.EXPRESSION):
           variable_ranks = _tensordeclaration(variable_ranks)
        return variable_ranks
    else:
        _error(TOKEN_ID.DECLARE.value)

def _tensordeclaration(variable_ranks):
    global _ident
    if _fits(TOKEN_ID.ALPHANUM) or _fits(TOKEN_ID.LOWERCASE_ALPHA):
        variablename = _ident
        _get_sym()
    else:
        _error('tensorname')
    if _fits(TOKEN_ID.NATNUM):
        rank = int(_ident)
        variable_ranks[variablename] = rank
        _get_sym()
    else:
        _error(TOKEN_ID.NATNUM.value)
    return variable_ranks

def _argument():
    global _desc
    global _ident
    if _fits(TOKEN_ID.DERIVATIVE):
        _get_sym()
        if _fits(TOKEN_ID.WRT):
            _get_sym()
            if _fits(TOKEN_ID.ALPHANUM) or _fits(TOKEN_ID.LOWERCASE_ALPHA):
                arg_name = _ident
            else:
                _error(TOKEN_ID.ALPHANUM.value)
        else:
            _error(TOKEN_ID.WRT.value)
    else:
        _error(TOKEN_ID.ARGUMENT.value)
    _get_sym()
    if not _desc == TOKEN_ID.NONE:
        raise Exception('Expected one argument to differentiate with respect to, but found multiple.')
    return arg_name

def _expressionpart():
    if _fits(TOKEN_ID.EXPRESSION):
        _get_sym()
        tree = _expr()
    else:
        _error(TOKEN_ID.EXPRESSION.value)
    return tree

def _expr():
    global _companion
    tree = _term()
    while _fits(TOKEN_ID.PLUS) or _fits(TOKEN_ID.MINUS):
        if _fits(TOKEN_ID.PLUS):
            _get_sym()
            tree = Tree(NODETYPE.SUM, '+', tree, _term(), companion=_companion)
        else:
            _get_sym()
            
            tree = Tree(NODETYPE.SUM, '+', tree, Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, _term(), companion=_companion), companion=_companion)
    return tree

def _term():
    global _companion
    tree = _factor()
    while _fits(TOKEN_ID.MULTIPLY) or _fits(TOKEN_ID.DIVIDE):
        if _fits(TOKEN_ID.MULTIPLY):
            _get_sym()
            if _fits(TOKEN_ID.LRBRACKET):
                _get_sym()
            else:
                _error(TOKEN_ID.LRBRACKET.value)
            leftIndices, rightIndices, resultIndices = _productindices()
            if _fits(TOKEN_ID.RRBRACKET):
                _get_sym()
            else:
                _error(TOKEN_ID.RRBRACKET.value)
            tree = Tree(NODETYPE.PRODUCT, f'*({leftIndices},{rightIndices}->{resultIndices})', tree, _factor(), companion=_companion)
            tree.set_indices(leftIndices, rightIndices, resultIndices)
        if _fits(TOKEN_ID.DIVIDE):
            _get_sym()
            tree = Tree(NODETYPE.PRODUCT, '_TO_BE_SET_ELEMENTWISE', tree, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, _factor(), companion=_companion), companion=_companion) # Indices will get set in set_tensorrank
    return tree

def _productindices():
    global _ident
    leftIndices = ''
    rightIndices = ''
    resultIndices = ''
    while _fits(TOKEN_ID.LOWERCASE_ALPHA):
        leftIndices = leftIndices + _ident
        _get_sym()
    if _fits(TOKEN_ID.COMMA):
        _get_sym()
    else:
        _error(TOKEN_ID.COMMA.value)
    while _fits(TOKEN_ID.LOWERCASE_ALPHA):
        rightIndices += _ident
        _get_sym()
    if _fits(TOKEN_ID.MINUS):
        _get_sym()
    else:
        _error(TOKEN_ID.MINUS.value)
    if _fits(TOKEN_ID.GREATER):
        _get_sym()
    else:
        _error(TOKEN_ID.GREATER.value)
    while _fits(TOKEN_ID.LOWERCASE_ALPHA):
        resultIndices += _ident
        _get_sym()
    return leftIndices, rightIndices, resultIndices

def _factor():
    global _companion
    parity = 0
    while _fits(TOKEN_ID.MINUS):
        parity = (parity+1) % 2
        _get_sym()
    if parity == 1:
        tree = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, _atom(), companion=_companion)
    else:
        tree = _atom()
    while _fits(TOKEN_ID.POW):
        _get_sym()
        if _fits(TOKEN_ID.LRBRACKET):
            _get_sym()
            tree = Tree(NODETYPE.POWER, '^', tree, _expr(), companion=_companion)
            if _fits(TOKEN_ID.RRBRACKET):
                _get_sym()
            else:
                _error(TOKEN_ID.RRBRACKET.value)
        else:
            tree = Tree(NODETYPE.POWER, '^', tree, _atom(), companion=_companion)
    return tree

def _atom():
    global _ident
    global _companion
    if _fits(TOKEN_ID.CONSTANT) or _fits(TOKEN_ID.NATNUM):
        tree = Tree(NODETYPE.CONSTANT, f'{_ident}_{_companion.new_constant()}', companion=_companion)
        _get_sym()
    elif _fits(TOKEN_ID.MINUS):
        _get_sym()
        if _fits(TOKEN_ID.CONSTANT) or _fits(TOKEN_ID.NATNUM):
            tree = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.CONSTANT, f'{_ident}_{_companion.new_constant()}', companion=_companion), companion=_companion)
            _get_sym()
        else:
            _error(TOKEN_ID.CONSTANT.value + ' or ' + TOKEN_ID.NATNUM.value)
    elif _fits(TOKEN_ID.ELEMENTWISE_FUNCTION):
        functionName = _ident
        _get_sym()
        if _fits(TOKEN_ID.LRBRACKET):
            _get_sym()
            tree = Tree(NODETYPE.ELEMENTWISE_FUNCTION, functionName, None, _expr(), companion=_companion)
            if _fits(TOKEN_ID.RRBRACKET):
                _get_sym()
            else:
                _error(TOKEN_ID.RRBRACKET.value)
        else:
            _error(TOKEN_ID.LRBRACKET.value)
    elif _fits(TOKEN_ID.SPECIAL_FUNCTION):
        functionName = _ident
        _get_sym()
        if _fits(TOKEN_ID.LRBRACKET):
            _get_sym()
            tree = Tree(NODETYPE.SPECIAL_FUNCTION, functionName, None, _expr(), companion=_companion)
            if _fits(TOKEN_ID.RRBRACKET):
                _get_sym()
            else:
                _error(TOKEN_ID.RRBRACKET.value)
        else:
            _error(TOKEN_ID.LRBRACKET.value)
    elif _fits(TOKEN_ID.ALPHANUM) or _fits(TOKEN_ID.LOWERCASE_ALPHA):
        if _ident == 'delta':
            _get_sym()
            if _fits(TOKEN_ID.LRBRACKET):
                _get_sym()
                if _fits(TOKEN_ID.NATNUM):
                    deltanum = int(_ident)
                    _get_sym()
                    if _fits(TOKEN_ID.RRBRACKET):
                        _get_sym()
                        tree = Tree(NODETYPE.DELTA, f'delta_{_companion.new_delta()}', companion=_companion)
                        tree.rank = 2*deltanum
                    else:
                        _error(TOKEN_ID.RRBRACKET.value)
                else:
                    _error(TOKEN_ID.NATNUM.value)
            else:
                _error(TOKEN_ID.LRBRACKET.value)
        else:
            tree = Tree(NODETYPE.VARIABLE, _ident, companion=_companion)
            _get_sym()
    elif _fits(TOKEN_ID.LRBRACKET):
        _get_sym()
        tree = _expr()
        if _fits(TOKEN_ID.RRBRACKET):
            _get_sym()
        else:
            _error(TOKEN_ID.RRBRACKET.value)
    else:
        _error(TOKEN_ID.CONSTANT.value + ' or ' + TOKEN_ID.ELEMENTWISE_FUNCTION.value + ' or ' + 'tensorname' +  ' or ' + TOKEN_ID.LRBRACKET.value)
    return tree

if __name__ == '__main__':
    example = 'declare A 2 expression delta(0) *(,ij->) A  derivative wrt a'
    dag, _, _ = parse(example)
    dag.dot('dags/dag')