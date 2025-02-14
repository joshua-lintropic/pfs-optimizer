import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def edge1():
    path = 'tests/edge1'

    N1 = 60
    N2 = 1000
    N = N1 + N2
    K = 100
    L = 10
    T_exp = 1
    u_max = np.array([10 for _ in range(N1)] + [0.5 for _ in range(N2)])
    T_target = np.array([10 for _ in range(N1)] + [1 for _ in range(N2)])
    return N, K, L, T_exp, u_max, T_target, path

def edge2():
    path = 'tests/edge2'

    N1 = 120
    N2 = 2000
    N = N1 + N2
    K = 100
    L = 25
    T_exp = 1
    u_max = np.array([10 for _ in range(N1)] + [0.5 for _ in range(N2)])
    T_target = np.array([10 for _ in range(N1)] + [1 for _ in range(N2)])
    return N, K, L, T_exp, u_max, T_target, path

def power_law():
    path = 'tests/power_law'

    # parameters
    N = 1000
    K = 100
    L = 10
    T_exp = 1

    # sample from pareto distribution
    scale = 100
    u_max = np.random.pareto(10, N) * scale + 1
    noise = np.random.normal(loc=1.0, scale=0.05, size=N)
    T_target = u_max / np.percentile(u_max, 99) * L * noise 

    for i in range(N):
        T_target[i] = max(0, min(L, int(T_target[i])))
    return N, K, L, T_exp, u_max, T_target, path

if __name__ == '__main__':
    # _, _, _, _, u_max, T_target = edge1()
    # print(u_max[:61])
    # print(T_target[:61])
    # print(T_target.shape)

    _, _, _, _, u_max, T_target = power_law()
    plt.hist(u_max, bins=10, alpha=0.5, label=r'$u_{\max}$')
    plt.savefig('power_umax.png')
    plt.clf()
    plt.hist(T_target, bins=10, alpha=0.5, label=r'$T_{\text{target}}$')
    plt.savefig('power_targets.png')
