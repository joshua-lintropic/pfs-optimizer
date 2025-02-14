import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys

import tester
from helpers import *

def simulate(data, start, stop, postgraph=True):
    """
    Runs the stochastic model to plan allocations of the Subaru Prime Focus Spectrograph.

    Conventions:
        N         : An integer representing the total number of galaxies. 
        K         : An integer representing the total number of fibers. 
        L         : An integer representing the total number of exposures.
        T_exp     : A float representing the time per exposure. 
        u_max     : An ndarray of shape (N,) storing maximum utilities per galaxy.
        T_target  : An ndarray of shape (N,) storing required time to realize the maximum
                    utility for each galaxy. (Must be same units as T_exp.)
        path      : The name of the output directory located under 'tests/'. 

    Parameters:
        data      : The attribute of `tester` used to import the testing data. 
        start     : An integer representing the starting exposure (1-indexed). 
        stop      : An integer representing the stopping exposure (1-indexed).
        postgraph : A boolean representing whether to overwrite old graphs.

    Returns:
        None (void). 
    """

    # generate input data
    N, K, L, T_exp, u_max, T_target, path = data()

    # track exposures-to-date
    if start == 1: 
        t = np.zeros((N, L))
        obs = np.zeros((L, K))
        log = open(f'{path}/log.txt', 'w')
    else: 
        t = np.load(f'{path}/t.npy')
        obs = np.load(f'{path}/obs.npy')
        log = open(f'{path}/log.txt', 'a')
    u_sharp = np.zeros((L,))

    # exposures are internally 0-indexed, but user-facing (graphs, input) are 1-indexed.
    for l in range(start-1, stop):
        log.write(f'=== EXPOSURE {l+1}: PROCESSING ===\n')

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
            log.write(f' -> lagrange multiplier: {ystar}\n')

        # determine the galaxies to observe
        if S == 1: mask = consume(u_max, R, K)
        else: mask = allocate(ystar, R, C, S, N, K)
        mask = sorted([x.item() for x in mask]) # TODO: find a better way to cast
        for i in mask: t[i, l] = 1

        # determine attained sharp utility
        for i in range(N):
            invested = np.sum(t[i, :])
            if invested == T_target[i]:
                u_sharp[l] += u_max[i]
            elif invested > T_target[i]:
                log.write(f'Warning, wasted on {i}: needed {T_target[i]}, used {invested}\n')

        # log the results
        obs[l, :] = np.where(t[:, l] == 1)[0]
        # count = len([x for x in mask if x < 120])
        log.write(f' -> fibers used: {len(obs[l, :])} / {K}\n')
        # log.write(f' -> expensive targets: {count} / {K}\n')
        log.write(f' -> galaxies observed: {mask}\n')
        log.write(f' -> sharp utility: {u_sharp[l]}\n')

        # save the progress
        np.save(f'{path}/t.npy', t)
        np.save(f'{path}/obs.npy', obs)

        log.write('\n')

    # plot max utility histograms
    graph = [[0] * K for _ in range(L)]
    for l in range(L):
        for k in range(K):
            graph[l][k] = int(obs[l, k].item()) # TODO: find a better way to cast
    ubins = np.linspace(start=0, stop=np.max(u_max), num=20)
    if stop == L and postgraph: start = 1
    for l in range(start-1, stop):
        plt.xlabel('Maximum Utility')
        plt.ylabel(f'Frequency ({K} fibers available)')
        plt.title(f'Maximum Utilities of Galaxies Observed (Exposure {l+1})')
        plt.hist(u_max[graph[l]], bins=ubins, label=f'Exposure {l+1}')
        plt.legend()
        plt.savefig(f'{path}/uhist{l+1}.png')

    # plot target time histograms
    plt.clf()
    tbins = np.linspace(start=1, stop=L, num=L)
    for l in range(start-1, stop):
        plt.xlabel('Target Time')
        plt.ylabel(f'Frequency ({K} Fibers Available)')
        plt.title(f'Target Times of Galaxies Observed (Exposure {l+1})')
        plt.hist(T_target[graph[l]], bins=tbins, label=f'Exposure {l+1}')
        plt.legend()
        plt.savefig(f'{path}/thist{l+1}.png')

    # graph attained sharp utility (all-or-nothing)
    plt.clf()
    plt.scatter(range(1, L+1), u_sharp, color='g')
    plt.xlabel(f'Exposure ({L} total)')
    plt.ylabel('Attained Sharp Utility')
    plt.title('Time Series of Attained Sharp Utility per Exposure')
    plt.savefig(f'{path}/sharp.png')

    log.close()

def main():
    try:
        argc = len(sys.argv)
        data = getattr(tester, sys.argv[1])
        if argc == 2:
            start = 1
            stop = data()[2]
        elif argc == 3:
            start = int(sys.argv[2])
            stop = data()[2]
        elif argc == 4:
            start = int(sys.argv[2])
            stop = int(sys.argv[3])
    except:
        print(f'Usage: python3 {sys.argv[0]} testname start stop')
        print(f'       (start and stop are optional.)')
        sys.exit(0)
    simulate(data=data, start=start, stop=stop, postgraph=True)

if __name__ == '__main__':
    main()
