import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import sys

from helpers import bracket
from tests import *

def simulate(data):
    # Generate input data
    N, K, L, T_e, u_max, T_target = data()
    print("### Successfully generated input data. ###\n")

    # Complete the optimization
    t = np.zeros((N, L)) # tracks exposures-to-date
    for l in range(L):
        # Determine remaining exposures necessary for each galaxy 
        print(f"=== EXPOSURE {l+1}: PROCESSING ===")
        T_prime = [] # time on galaxy i so far
        if l == 0: T_prime = np.array([0 for i in range(N)])
        if l > 0: T_prime = np.array([np.sum(t[i, :l]) for i in range(N)])

        R = np.ceil(np.maximum((T_target - T_prime) / T_e, 0)).astype(int) # remaining exposures needed
        
        # Separate the coefficients
        C = np.array([u_max[i] * sp.special.comb(L - l, R[i]) for i in range(N)])
        S = L - l # how many possible exposures left
        def p(i: int, y) -> float: # p -> theta, y -> lambda
            # Evaluate degenerate cases
            if S == 1: return int(R[i] == 1)
            if R[i] == 0: return 0
            if R[i] == S:
                obj = lambda p : -C[i] * p**S + y*p
                args = np.array([0, 1, (y/C[i]/S)**(1/(S-1))])
                return args[np.argmin(obj(args))]
            if R[i] > S: return 0 # impossible to attain max value
            # Determine the analytic coefficients
            a = np.zeros((S,))
            a[0] += y
            a[R[i]-1] += -C[i] * R[i]
            a[S-1] += C[i] * S * (-1)**(S-R[i]-1)
            for k in range(1, S-R[i]):
                a[k+R[i]-1] += C[i] * (-1)**(k-1) * (S*sp.special.comb(S-R[i]-1, k-1) + R[i]*sp.special.comb(S-R[i]-1, k))
            # Find the minimizer
            endpoints = np.array([0, 1])
            roots = np.polynomial.polynomial.polyroots(a)
            args = np.concatenate([endpoints, roots[(0 <= roots) & (roots <= 1)]])
            obj = lambda p : - C[i] * p**R[i] * (1 - p)**(S - R[i]) + y * p
            return args[np.argmin(obj(args))]

        F = lambda y : np.sum(np.array([p(i, y) for i in range(N)])) - K

        # === Bracket: find negative and positive values === 
        try:
            pos_t, neg_t = bracket(F, max_iter=10000)
            ystar, result = sp.optimize.brentq(F, neg_t[0], pos_t[0], full_output=True)
            if not result.converged:
                raise RuntimeError(f"Brentq did not converge.")
        except ValueError:
            if S != 1:
                result = sp.optimize.minimize(lambda y : F(y)**2, 0.8, method='BFGS')
                ystar = result.x[0]
                if not result.success: raise RuntimeError("BFGS did nto converge.")
            else: 
                ystar = 0

        print(f"Found optimal Lagrange multiplier: {ystar}")

        mask = np.array([])
        if S != 1: 
            allocations = np.array([p(i, ystar) for i in range(N)])
            mask = (-allocations).argsort()[:K]
        else:
            alive = np.where(R == 1)[0]
            greedy = np.argsort(u_max)[::-1]
            count = 0
            for idx in greedy:
                if idx not in alive: continue
                mask = np.append(mask, idx)
                count += 1
                if count == K: break

        # Allocate the next available fiber to each selected galaxy
        u_selected = []
        fiber = 0
        for i in range(N):
            t[i, l] = 0
            if i in mask: 
                t[i, l] = 1
                fiber += 1
                u_selected += [u_max[i]]
        print(f"{fiber} / {K} fibers used")
        plt.hist(u_selected, bins=10, density=True)
        plt.savefig(f"hist{l}.png")
        # np.save('allocations.npy', t)

        print()

    # Print attained sharp utility (all-or-nothing)
    u_sharp = 0
    u_attained = []
    for i in range(N):
        if np.sum(t[i, :]) == T_target[i]:
            u_sharp += np.sum(u_max[i])
            u_attained += [u_max[i]]
        elif np.sum(t[i, :]) > T_target[i]:
            printf(f"wasteful on target {i}")
    print(f"Sharp Utility: {u_sharp}")
    plt.hist(u_attained, bins=10, density=True)
    print('max attained u_max', max(u_attained))
    plt.savefig('hist.png')

def main():
    simulate(data=power_law)

if __name__ == "__main__":
    main()
