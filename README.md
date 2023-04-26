# tensor_calc
A tool for symbolically differentiating tensor expressions, created for by bachelor's thesis at FSU Jena in the summer semester of 2022.

## Usage
View `demo.py` to get an idea of how the tool can be used to differentiate a tensor expression and how the resulting derivative can be checked numerically for correctness.

## Language
The tool takes a string consisting of three parts as input:
- the declaration part specifies the tensor variables used in the expression, along with how many axes they each have
- the expression part specifies the tensor expression to be differentiated
- the argument part specifies with respect to which tensor variable the expression is to be differentiated

The concrete language is specified by the following grammar in extended Backus-Naur form:
```
input = declaration expressionpart argument

declaration = 'declare' (tensordeclaration)*
tensordeclaration = tensorname (digit)*
tensorname = alpha { alpha | digit }
alpha = smallalpha | largealpha
smallalpha = [a-z]
largealpha = [A-Z]

expressionpart = 'expression' expr
expr = term (( '+' | '-' ) term)*
term = factor (( '*(' productindices ')') factor | '/' factor)*
productindices = tensorindices ',' tensorindices '->' tensorindices
tensorindices = (smallalpha)*
factor = {'-'} atom ('^' ('(' expr ')' | atom))*
atom = number | function '(' expr ')' | tensorname | '(' expr ')' | delta
number = ['-'] digit* '.' digit* [( 'e' | 'E' ) ['+' | '-'] (digit)*]
digit = [0-9]
function = 'sin' | 'cos' | 'tan' | 'arcsin' | 'arccos' | 'arctan' | 'tanh' | 'exp' 
         | 'log' | 'sign' | 'relu' | 'abs' | 'det' | 'inv' | 'adj'
delta = 'delta(' {digit}+ ')' 

argument = 'derivative wrt' tensorname
```
