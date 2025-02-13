import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def normal():
    N = 10000
    K = 2400
    L = 10
    u_max = np.random.normal(loc=100, scale=10, size=(N,)) # max scientific utility
    T_target = np.random.normal(loc=L/2, scale=L/5, size=(N,)) # target time to reach max utility
    T_e = 1 # time per exposure

    fig, ax = plt.subplots(ncols=2)
    fig.set_figwidth(20)
    fig.suptitle("Max Observation Utility & Target Time")

    ax[0].plot(u_max, color='blue')
    ax[0].set_xlabel("Galaxy Index")
    ax[0].set_ylabel("Maximum Observation Utility")

    ax[1].plot(T_target, color='green')
    ax[1].set_xlabel("Galaxy Index")
    ax[1].set_ylabel("Target Time to Reach Max Utility (hours)")
    fig.savefig("data.png")

    return N, K, L, T_e, u_max, T_target

def edge1():
    N1 = 60
    N2 = 1000
    N = N1 + N2
    K = 100
    L = 10
    T_e = 1
    u_max = np.array([10 for _ in range(N1)] + [0.5 for _ in range(N2)])
    T_target = np.array([10 for _ in range(N1)] + [1 for _ in range(N2)])
    return N, K, L, T_e, u_max, T_target

def power_law():
    # parameters
    N = 1000
    K = 100
    L = 10
    T_e = 1

    # sample from pareto distribution
    scale = 100
    u_max = np.random.pareto(10, N) * scale + 1
    noise = np.random.normal(loc=1.0, scale=0.05, size=N)
    T_target = u_max / np.percentile(u_max, 99) * L * noise 
    # TODO: skew the distribution
    for i in range(N):
        T_target[i] = max(0, min(L, int(T_target[i])))
    return N, K, L, T_e, u_max, T_target

if __name__ == "__main__":
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
