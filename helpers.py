import numpy as np
import scipy as sp

def probablize(i, y, R_i, C_i, S):
    """
    Find the probability term associated with the objective for a particular galaxy.

    Parameters:
        i   : The index of the relevant galaxy. 
        y   : The Lagrange multiplier on the sum constraint.
        R_i : The number of remaining exposures to reach i's target time. 
        C_i : The binomial coefficient associated with the expansion.

    Returns:
        A float which is the optimal Lagrange multiplier on the sum constraint.
    """

    # Evaluate degenerate cases
    if S == 1: 
        return int(R_i == 1)
    if R_i == 0: 
        return 0
    if R_i == S:
        obj = lambda x : -C_i * x**S + y*x
        args = np.array([0, 1, (y/C_i/S)**(1/(S-1))])
        return args[np.argmin(obj(args))]
    if R_i > S: 
        return 0

    # Determine the analytic coefficients
    a = np.zeros((S,))
    a[0] += y
    a[R_i-1] += -C_i * R_i
    a[S-1] += C_i * S * (-1)**(S-R_i-1)
    for k in range(1, S-R_i):
        comb1 = sp.special.comb(S - R_i - 1, k - 1)
        comb2 = sp.special.comb(S - R_i - 1, k)
        a[k + R_i - 1] += C_i * (-1)**(k-1) * (S * comb1 + R_i * comb2)

    # Find the minimizer
    endpoints = np.array([0, 1])
    roots = np.polynomial.polynomial.polyroots(a)
    args = np.concatenate([endpoints, roots[(0 <= roots) & (roots <= 1)]])
    obj = lambda x : - C_i * x**R_i * (1 - x)**(S - R_i) + y * x

    return args[np.argmin(obj(args))]


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

    # evaluate the function at the initial point
    try:
        f_initial = f(initial)
    except Exception as e:
        raise ValueError('Error evaluating function at initial value: {}'.format(e))
    
    # check which sign we need to search for
    if f_initial > 0:
        pos_candidate = (initial, f_initial)
    elif f_initial < 0:
        neg_candidate = (initial, f_initial)

    # first attempt: linear search both directions
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

    # second attempt: exponential search in both directions
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
    
    # unable to bracket, report error
    raise ValueError('Could not find both positive and negative values for the given function.')

def dualize(res):
    try:
        pos_t, neg_t = bracket(res, max_iter=10000)
        ystar, result = sp.optimize.brentq(res, neg_t[0], pos_t[0], full_output=True)
        if not result.converged:
            raise RuntimeError(f'Brentq did not converge.')
    except ValueError:
        result = sp.optimize.minimize(lambda y : res(y)**2, 0.8, method='BFGS')
        ystar = result.x[0]
        if not result.success: 
            raise RuntimeError('BFGS did not converge.')

    return ystar

def allocate(ystar, R, C, S, N, K):
    """
    Selects galaxies to observe by rank-ordering on the stochastic objective. 

    Parameters:
        ystar : The optimal Lagrange multiplier on the sum constraint.
        R     : A numpy array of remaining exposures needed.
        C     : A numpy array of binomial coefficients. 
        S     : The number of possible exposures remaining.
        N     : The total number of galaxies. 
        K     : The number of available fibers. 

    Returns:
        An ndarray of shape (K,) with galaxies to observe.
    """

    allocations = [probablize(i, ystar, R[i], C[i], S) for i in range(N)]
    mask = (-np.array(allocations)).argsort()[:K]

    return mask.tolist()

def consume(u_max, R, K):
    """
    Selects galaxies to observe by rank-ordering all galaxies which can be
    completed in one exposure. 

    Parameters:
        u_max : A numpy array of maximum utilities from acquiring the target time.
        R     : A numpy array of remaining exposures needed.
        K     : The number of available fibers. 

    Returns:
        An ndarray of shape (K,) with galaxies to observe. 
    """

    alive = np.where(R == 1)[0]
    greedy = np.argsort(u_max)[::-1]
    mask = []

    for idx in greedy:
        if idx not in alive: continue
        mask += [idx]
        if len(mask) == K: break

    return [x.item() for x in mask]

if __name__ == '__main__':
    test_function = lambda x : np.sin(x) - 0.5

    try:
        (x_pos, f_pos), (x_neg, f_neg) = find_sign_values(test_function, initial=0.0, step=0.1)
        print('Found f(x) > 0 at x = {:.5f} (f(x) = {:.5f})'.format(x_pos, f_pos))
        print('Found f(x) < 0 at x = {:.5f} (f(x) = {:.5f})'.format(x_neg, f_neg))
    except ValueError as err:
        print(err)
