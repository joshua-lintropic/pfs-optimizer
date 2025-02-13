# TODO
fixed bins
target times, u_max at every exposure
sum of total attained utilities as function of exposure
double total time on toy case (or 25 exposures), double Ns
make longest runs < total time

# pfs-optimizer

(Idea is from Professor Peter Melchior)
# PFS Per-Exposure Targeting Optimization
We assume that we have $i=1,\dots,N$ galaxies, each of which with a predefined utility $u_i$. The utilities can be parameterized as $u_i=\hat u_i\ \sigma((T_i-\hat T_i)/\delta T)$ with a maximum utility $\hat u$, which is modulated by a sigmoid function, to create the utility gains only after a desired integration time $\hat T$ has been reaching. For future reference, $\hat u$ and $\hat T$ may be grouped into classes of targets, not individual ones.

The observations are given by $L$ exposures of equal length $T_e$ in time on an instrument with $K=2400$ fibers. We want to find per-exposure assignments $t_{ikl}$ to

$$
\begin{align}
\text{maximize} &\sum_{i=1}^N u_i = \sum_i \hat u_i\ \sigma((\sum_{k,l} t_{ikl} -\hat T_i)/\delta T)\\
\text{subject\ to}\ & \forall l: \sum_{k,i} t_{ikl} = K\\
&\forall k,l: \sum_i t_{ikl} \leq 1\\
&\forall i,l: \sum_k t_{ikl} \leq 1\\
&\forall i,k: t_{ikl} \in \{0,1\}
\end{align}
$$
This is combinatorially really hard and practically impossible when only next-exposure decision can be made. In this case, we have some $l\in[1,L]$, and we want to find the best assignments for the next exposure  $l=1,\dots,L$ such that the solutions approximates the global assignment/scheduling solution. We therefore introduce  "intermediate" utilities $u_{il}$ and then determine the targeting assignments $s_{il}$ for the exposure $l$ such that they

$$
\begin{align}
\text{maximize} &\sum_{i=1}^N u_{il} = \sum_i \hat u_i\, s_{il}\\
\text{subject\ to}\ &\sum_{i} s_{il} = K\\
&\forall i: s_{il} \in \{0,1\}
\end{align}
$$
But that doesn't have any aspect of long-term planning, so this approach goes after the high-utility targets whenever it can (not a terrible policy, but not very equitable). Let's see if we can find some more useful structure in lieu of the $s_{il}$.

The idea is to make this problem probabilistic and continuous and

$$
\begin{align}
\text{maximize} &\sum_{i=1}^N u_{il} = \sum_i \hat u_i\, \text{Pr}(\sum_{k,l'} t_{ikl'} = \hat T_i\mid \theta_{il})\\
\text{subject\ to}\ &\sum_{i} \theta_{il} = K\\
&\forall i: 0\leq\theta_{il}\leq 1
\end{align}
$$

where $\text{Pr}(\sum_{k,l'} t_{ikl'} = \hat T_i\mid\theta_{il})=\text{Binom}(L-l, (\hat T_i - T'_i)/T_e, \theta_{il})$ is the probability of getting the desired total integration by the end of the program. This term provides memory because it includes integration from earlier exposures $T'_i=T_e\sum_{l'=1}^l t_{ikl'}$ and future outlook because it counts the number of choices available to get the remaining exposures.

By optimizing the targeting probabilities $\theta_{il}$ we thus strike a balance between high-utility  and feasible targets. Since we need to make targeting choices, we would rank-order the targets $i$ by their contribution to eq. (9), and pick the top $K$.

What remains is the choice of the $\theta_{il}$. If all targets are equally easy to get, the initial probabilities are $\forall i:\theta_{i1}=K/N$. This would pick all possible high-utility targets first, no optimization needed. We should test this as a baseline, but it is likely not close to optimal because it sacrifices all lower-utility targets even if there are so many to have that they have more aggregate utility.

One can think of several ways of determining "better" $\theta_{il}$:
* Brute-force optimization  possible given $N$ is $\mathcal{O}(10^4)$, but likely to get stuck in local minima.
* Introduce classes such that $\theta_{il}\propto N_c K/N$ for $i\in\mathcal{C}_c$, which aims to get $N_c$ total objects in class ${C}_c$ after all exposures are done.
* Minimizing the variance of estimator, which prefers $\theta_{il}\in\{0,1\}$.
* Reinforcement learning: Simulate the policy to find q table as function of $\hat u_i, l, T'_i$ and potentially other terms for class-based utilities.
