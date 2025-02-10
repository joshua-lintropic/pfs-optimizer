# Key Idea
The point is that instead of looking at 

$$
\begin{aligned}
\text{minimize} \quad & -\sum_{i=1}^{N} \hat{u}_{i} {L - l \choose R_{i}} \theta_{i}^{R_{i}}(1 - \theta_{i})^{L - l - R_{i}} \\
\text{subject to} \quad & 0 \leq \theta_{i} \leq 1, \quad \sum_{i=1}^{N} \theta_{i} = K
\end{aligned}
$$

which is an $N=10,000$-dimensional nonconvex optimization problem, instead we first formulate the Lagrangian

$$
\mathcal{L}(\theta_{i}, \lambda) = -\sum_{i=1}^{N} \hat{u}_{i} {L - l \choose R_{i}} \theta_{i}^{R_{i}}(1 - \theta_{i})^{L - l - R_{i}} + \lambda \left( \sum_{i=1}^{N} \theta_{i} - K \right) 
$$

Then we solve the problem 

$$
\begin{aligned}
\text{minimize} \quad & - C_{i}\theta_{i}^{R_{i}}(1 - \theta_{i})^{S - R_{i}} + \lambda \theta_{i} \\
\text{subject to} \quad & 0_{i} \leq \theta \leq 1
\end{aligned}
$$

which is smooth over compact domain and thus we simply need to solve a low-degree polynomial (since $L$ is the number of exposures and thus the degree is not too large) by checking endpoints and local extrema. Doing so allows us to construct a set of functions $\theta_{i}(\lambda)$ giving the optimal $\theta_{i}$. Then to enforce $\sum_{i=1}^{N} \theta_{i} = K$, all we must do is solve 

$$
\Theta(\lambda) = \sum_{i=1}^{N} \theta_{i}(\lambda) = K
$$

which can be done by bisection or other standard scalar algorithms.

# Analytical Form of Polynomial
To solve a problem of form
$$
\begin{aligned}
\text{minimize} \quad & f(\theta) = - C\theta^{R}(1 - \theta)^{S - R} + \lambda \theta \\
\text{subject to} \quad & 0 \leq \theta \leq 1
\end{aligned}
$$
we must analytically express the gradient as a polynomial: for $S \geq 2$ and $1\leq R\leq S-1$, 

$$
\begin{aligned}
0 &= \frac{df}{d\theta} = C \left[ -R\theta^{R-1}(1-\theta)^{S-R}+(S-R) \theta^R(1-\theta)^{S-R-1} \right] + \lambda \\
&= C \theta^{R-1}(1-\theta)^{S-R-1} \left[ -R(1-\theta) + (S-R)\theta \right] + \lambda \\
&= C \theta^{R-1}(1-\theta)^{S-R-1} (S\theta-R) + \lambda
\end{aligned}
$$

Expanding binomial coefficients, 

$$
\begin{aligned}
0 &= C \theta^{R-1}(S\theta-R) \left[ \sum_{k=0}^{S-R-1} (-1)^k {S-R-1 \choose k} \theta^k \right] + \lambda \\
&= C \theta^{R-1} \left[ S \sum_{k=0}^{S-R-1} (-1)^k {S-R-1 \choose k} \theta^{k+1} - R \sum_{k=0}^{S-R-1} (-1)^k {S-R-1 \choose k} \theta^k \right] + \lambda \\
&= C \theta^{R-1} \left[ \sum_{k=1}^{S-R} (-1)^{k-1} {S-R-1 \choose k-1} \theta^k - R \sum_{k=0}^{S-R-1} (-1)^k {S-R-1 \choose k} \theta^k \right] + \lambda\\
&= C \theta^{R-1} \left[ S(-1)^{S-R-1} \theta^{S-R} + \sum_{k=1}^{S-R-1} (-1)^{k-1}\left( S {S-R-1 \choose k-1} + R {S-R-1 \choose k} \right) \theta^k - R \right] + \lambda
\end{aligned}
$$

from which the optimal value $\theta^*$ may be numerically solved (for any given $\lambda$).

Meanwhile if $S=1$ (recall $S \neq 0$ since it is the number of remaining exposures), then recall that the objective is for $X \sim\text{Binom}(S,\theta)$ then just 

$$
\begin{cases}
-\hat{u} (1-\theta) + \lambda & R=0 \\
-\hat{u} \theta + \lambda & R=1 \\
\lambda & \text{otherwise}
\end{cases}
$$

Of course since the minimization occurs over $[0, 1]$ (and looking forward to the sum constraint on $\theta$) this is solved by 

$$
\theta^* = \begin{cases}
1 & R=1 \\
0 & \text{otherwise}
\end{cases}
$$


Finally consider $S>1$, but $R=0$ or $R=S$. If $R=0$, then again $\theta^*=0$. If $R=S$, then 

$$
\begin{aligned}
\text{minimize} \quad & -C \theta^S + \lambda \theta \\
\text{subject to} \quad & 0 \leq \theta \leq 1
\end{aligned}
$$

reduces to testing $\theta\in\{0, 1, \sqrt[S-1]{\lambda/CS}\}$ from first-order conditions.