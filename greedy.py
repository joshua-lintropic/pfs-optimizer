import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import sys

import tester
from helpers import *

def optimize(data, start, stop, postgraph=True):
    # TODO: update docs
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
    t = np.zeros((N, L))
    u_sharp = np.zeros((L,))
    waste = np.zeros((N,))

    # exposures are internally 0-indexed, but user-facing (graphs, input) are 1-indexed.
    for l in range(start-1, stop):
        print(f'=== EXPOSURE {l+1}: PROCESSING ===')

        # determine remaining exposures necessary for each galaxy 
        if l == 0: T_past = np.array([0 for i in range(N)])
        else: T_past = np.array([np.sum(t[i, :l]) for i in range(N)])
        R = np.ceil(np.maximum((T_target - T_past) / T_exp, 0)).astype(int)
        
        # calculate the remaining exposures possiblbe
        S = L - l

        # choose the highest utility-per-time ratio
        def score(i):
            alpha = (l+1) / L
            if R[i] == 0 or R[i] > S: return 0
            return (u_max[i] / R[i]) * (T_target[i] / sum(T_target))**(np.arcsin(1 - alpha))
        mask = sorted(list(range(N)), key=score, reverse=True)[:K]
        for i in mask: t[i, l] = 1

        # determine attained sharp utility
        for i in range(N):
            invested = np.sum(t[i, :])
            if invested == T_target[i]:
                u_sharp[l] += u_max[i]
            elif invested > T_target[i]:
                # print(f'Warning, wasted on {i}: needed {T_target[i]}, used {invested}\n')
                pass

        print(f'Sharp Utility: {u_sharp[-1]}')
        print()

    # print waste
    for i in range(N):
        invested = np.sum(t[i, :])
        waste[i] += max(invested - T_target[i], 0)
    print(f'Total Waste: {np.sum(waste)}')

    # graph attained sharp utility (all-or-nothing)
    plt.scatter(range(1, L+1), u_sharp, color='g')
    plt.xlabel(f'Exposure ({L} total)')
    plt.ylabel('Attained Sharp Utility')
    plt.title('Time Series of Attained Sharp Utility per Exposure')

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
    optimize(data=data, start=start, stop=stop, postgraph=True)

if __name__ == '__main__':
    main()
