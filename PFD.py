import sympy as sp

def partial_fraction_decomposition(expr):
    """
    Decompose a given expression of the form 1/(polynomial) into its partial fraction form.
    This version handles more complex polynomial expressions in the denominator.

    Args:
    expr (str): The polynomial in the denominator in a more flexible format.

    Returns:
    str: The partial fraction decomposition of 1/expr.
    """
    x = sp.symbols('x')
    # Evaluating the expression within the sympy environment
    polynomial = sp.sympify(expr, evaluate=True)
    decomposed_expr = sp.apart(1 / polynomial, x)
    return str(decomposed_expr)

# Testing with the new format
example_expr = "(x-a)*(x-(a + b*i))*(x- (a - b*i))*(x-(a + c*i))*(x- (a - c*i))"
decomposed= partial_fraction_decomposition(example_expr)
print(decomposed)
