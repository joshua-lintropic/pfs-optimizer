import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import tests

from helpers import *

def simulate(data, preload=False):
    # generate input data
    N, K, L, T_exp, u_max, T_target, path = data()
    print('=== Successfully generated input data. ===\n')

    # track exposures-to-date
    t = np.zeros((N, L))
    u_sharp = np.zeros((L,))
    bins = np.linspace(start=0, stop=np.max(u_max), num=20)

    for l in range(L):
        print(f'=== EXPOSURE {l+1}: PROCESSING ===')

        # determine remaining exposures necessary for each galaxy 
        if l == 0: T_past = np.array([0 for i in range(N)])
        else: T_past = np.array([np.sum(t[i, :l]) for i in range(N)])
        R = np.ceil(np.maximum((T_target - T_past) / T_exp, 0)).astype(int)
        
        # calculate the (parallelized) coefficients
        C = np.array([u_max[i] * sp.special.comb(L - l, R[i]) for i in range(N)])

        # calculate the remaining exposures possiblbe
        S = L - l

        # calculate residual on sum constraint
        res = lambda y : np.sum(np.array([probablize(i, y, R[i], C[i], S) for i in range(N)])) - K

        # bracket the residual function to find optimal Lagrange multiplier
        if S != 1: 
            ystar = dualize(res)
            print(f' -> lagrange multiplier: {ystar}')

        # determine the galaxies to observe
        if S == 1: mask = consume(u_max, R, K)
        else: mask = allocate(ystar, R, C, S, N, K)
        mask = sorted([x.item() for x in mask])
        for i in mask: t[i, l] = 1

        # determine attained sharp utility
        attained = 0
        for i in range(N):
            if np.sum(t[i, :]) == T_target[i]:
                attained += u_max[i]
            elif np.sum(t[i, :]) > T_target[i]:
                print(f'wasteful on target {i}')
        u_sharp[l] = attained

        # diagnose the results
        observed = np.where(t[:, l] == 1)[0]
        print(f' -> fibers used: {len(observed)} / {K}')
        print(f' -> galaxies observed:', mask)

        # graph the histograms
        plt.xlabel('maximum utility')
        plt.ylabel(f'frequency ({K} fibers available)')
        plt.title(f'max utilities of galaxies observed during exposure {l}')
        plt.hist(u_max[observed], bins=bins, label=f'exposure {l}')
        plt.legend()
        plt.savefig(f'{path}/hist{l}.png')

        # save the progress
        np.save(f'{path}/t.npy', t)

        print()

    # Print attained sharp utility (all-or-nothing)
    plt.clf()
    plt.hist(u_sharp, bins=10, alpha=0.5)
    plt.savefig(f'{path}/hist.png')

def main():
    simulate(data=getattr(tests, sys.argv[1]), preload=False)

if __name__ == '__main__':
    main()
