import numpy as np

def bracket(f, initial=0.0, step=1.0, max_iter=10000):
    """
    Search for two points x1 and x2 such that f(x1) > 0 and f(x2) < 0.
    
    The search begins at 'initial' and expands outwards in both the positive and
    negative directions in increments of 'step'. If that fails to find both signs,
    the algorithm then tries an exponential expansion of the search interval.
    
    Parameters:
        f        : A callable function f(x) that returns a numerical value.
        initial  : The starting x-value for the search (default is 0.0).
        step     : The linear step size to use in the initial search (default is 1.0).
        max_iter : Maximum number of iterations in the linear (or exponential) search.
        
    Returns:
        A tuple ((x_pos, f(x_pos)), (x_neg, f(x_neg))) where:
          - f(x_pos) > 0,
          - f(x_neg) < 0.
          
    Raises:
        ValueError: If no such points can be found within the search limits.
    """
    pos_candidate = None
    neg_candidate = None

    # Evaluate the function at the initial point.
    try:
        f_initial = f(initial)
    except Exception as e:
        raise ValueError("Error evaluating function at initial value: {}".format(e))
    
    # Check the initial value.
    if f_initial > 0:
        pos_candidate = (initial, f_initial)
    elif f_initial < 0:
        neg_candidate = (initial, f_initial)
    # Note: if f(initial)==0, we treat it as neither positive nor negative
    # (but you might decide to take it as one or both, depending on your needs)

    # First attempt: linear search in both directions.
    for i in range(1, max_iter + 1):
        for direction in [+1, -1]:
            x_candidate = initial + direction * i * step
            try:
                f_val = f(x_candidate)
            except Exception:
                # If evaluation fails at this point, skip it.
                continue
            if f_val > 0 and pos_candidate is None:
                pos_candidate = (x_candidate, f_val)
            elif f_val < 0 and neg_candidate is None:
                neg_candidate = (x_candidate, f_val)
            if pos_candidate is not None and neg_candidate is not None:
                return pos_candidate, neg_candidate

    # If we still haven't found both a positive and a negative value, try an exponential expansion.
    expo_step = step
    for i in range(max_iter):
        for direction in [+1, -1]:
            x_candidate = initial + direction * expo_step
            try:
                f_val = f(x_candidate)
            except Exception:
                continue
            if f_val > 0 and pos_candidate is None:
                pos_candidate = (x_candidate, f_val)
            elif f_val < 0 and neg_candidate is None:
                neg_candidate = (x_candidate, f_val)
            if pos_candidate is not None and neg_candidate is not None:
                return pos_candidate, neg_candidate
        expo_step *= 2  # Double the search distance.
    
    # If we still have not found both signs, report an error.
    raise ValueError("Could not find both positive and negative values for the given function.")

# ---------------------------------------
# Example usage:
if __name__ == '__main__':
    import math

    # Define a test function. For example, f(x) = sin(x) - 0.5 has both positive and negative values.
    def test_function(x):
        return math.sin(x) - 0.5

    try:
        (x_pos, f_pos), (x_neg, f_neg) = find_sign_values(test_function, initial=0.0, step=0.1)
        print("Found f(x) > 0 at x = {:.5f} (f(x) = {:.5f})".format(x_pos, f_pos))
        print("Found f(x) < 0 at x = {:.5f} (f(x) = {:.5f})".format(x_neg, f_neg))
    except ValueError as err:
        print(err)

# def bracket(F, y0=0.0, factor=2.0, max_iters=100):
#     """
#     Bracket the root of F by moving in the correct direction.
#     Returns (y_low, y_high) such that F(y_low) and F(y_high) have opposite signs.
#     """
#     f0 = F(y0)
#     if f0 == 0:
#         return y0, y0
# 
#     # Decide which direction to search:
#     # if y0 > 0: We need to increase y until F becomes negative.
#     y_low = y0
#     y_high = y0 + 1e-6  # small increment to start
#     iter_count = 0
#     while F(y_high) > 0 and iter_count < max_iters:
#         y_high *= factor
#         iter_count += 1
#     # if F(y_high) > 0:
#     #     raise ValueError("Unable to bracket the root in the upward direction.")
#     if F(y_low) > 0 and F(y_high) <= 0:
#         return y_low, y_high
# 
#     # if f0 < 0: we need to decrease y until F becomes positive.
#     y_high = y0
#     y_low = y0 - 1e-6  # small decrement to start
#     iter_count = 0
#     while F(y_low) < 0 and iter_count < max_iters:
#         # For negative numbers, multiplying by factor preserves sign if y0 is nonzero.
#         # You might use y_low = y_low * factor (if y0 is negative) or y_low -= some_delta.
#         y_low *= factor  # works if y0 is negative
#         iter_count += 1
#     if F(y_high) < 0 and F(y_low) >= 0:
#         return y_low, y_high
#     raise ValueError("Unable to bracket")

def bisect(F, tol=1e-6):
    # Start at y=0 and bracket the root.
    y_low, y_high = find_y_bracket(F, y0=0.0)
    
    # Use bisection (or you can use scipy.optimize.brentq) to find the root.
    while (y_high - y_low) > tol:
        y_mid = 0.5 * (y_low + y_high)
        f_mid = F(y_mid)
        if f_mid == 0:
            return y_mid
        # Since F is monotonic, decide which half-interval contains the sign change.
        if f_mid > 0:
            # Root must be to the right.
            y_low = y_mid
        else:
            y_high = y_mid
    return 0.5 * (y_low + y_high)

