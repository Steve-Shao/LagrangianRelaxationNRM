# Implementation of the Lagrangian Relaxation Algorithm for Network Revenue Management

<br>



<!--
The abstract should address:
- Who is the customer? What is their need? Why do they need this solution?
- What does this product do? How does it benefit the customer?
- How is success measured for this product? When will it be considered successful?
-->

This repository provides a public implementation of the Lagrangian Relaxation Algorithm for network revenue management, originally developed by Prof. Huseyin Topaloglu (2009). 
Researchers in operations research often use this algorithm and its test dataset as benchmarks, but no open implementation has been available until now. 
This code allows researchers to test, compare, and build upon the algorithm easily. 
We have tested our implementation on all instances in Prof. Topaloglu's dataset, and the results match those reported in his paper. 
This repository aims to support reproducible research and further development in network revenue management.

<br>



<!--
The introduction should summarize:
-->

## About

...

<br>

<!--
The literature review should address:
-->

## Resources

- 

<br>



<!--
The formulation review should address:
-->

## The NRM Problem

Consider a **network revenue management problem** with:
- A set of resources (flight legs) $\mathcal{L}$, each with capacity $c_i$ for $i \in \mathcal{L}$.
- A set of products (itineraries) $\mathcal{J}$, each with revenue $f_j$ for $j \in \mathcal{J}$.
    - Each purchase of product $j$ consumes $a_{ij}$ units of capacity from resource $i$ for each $i$.
- Discrete time horizon $\mathcal{T} = \{1, \ldots, \tau\}$.

In each period $t$:
- **At most one customer arrives**
- The customer requests product $j$ with probability $p_{jt}$
- $\sum_{j \in \mathcal{J}} p_{jt} \le 1$

Note: By adding a dummy itinerary $\psi$ with

$$
\begin{align*}
    f_{\psi} &= 0 \\
    a_{i\psi} &= 0 \quad \forall i \in \mathcal{L} \\
    p_{\psi t} &= 1 - \sum_{j \in \mathcal{J}} p_{jt} \quad \forall t \in \mathcal{T}
\end{align*}
$$

Thus, we can assume that in each period $t$:
- One customer arrives
- The customer requests product $j$ with probability $p_{jt}$
- $\sum_{j \in \mathcal{J}} p_{jt} = 1$

Let $x_{it}$ be the remaining capacity of resource $i$ at the start of period $t$.
Let $x_t = (x_{1t}, x_{2t}, \dots, x_{|\mathcal{L}|, t})$ be the state vector.
Let

$$
\begin{align*}
    C &= \max_{i \in \mathcal{L}} c_i \\
    \mathcal{C} &= \{0, 1, \ldots, C\}
\end{align*}
$$

and let $\mathcal{C}^{|\mathcal{L}|}$ be the state space.

<br>



<!--
The method should address:
-->

## The Algorithm

### Dynamic Programming Formulation

- Let $u_{jt} \in \{0,1\}$ indicate whether to accept (1) or reject (0) a request for product $j$.
- Let $V_t(x_t)$ be the maximum expected revenue from period $t$ to $\tau$ given capacities $x_t$:

\[
\begin{align*}
V_t(x_t) = \max_{u_t \in \mathcal{U}(x_t)} 
    \left\{ \sum_{j\in \mathcal{J}} p_{jt} 
    \left[ 
        f_j u_{jt} + 
        V_{t+1} \left(x_t - u_{jt}\sum_{i\in \mathcal{L}}a_{ij}e_i\right) 
    \right] \right\}
\tag{DP1}
\end{align*}
\]

where

$$
\mathcal{U}(x_t) = \left\{ 
    u_{t} \in \{0,1\}^{|\mathcal{J}|} : 
    a_{ij} u_{jt} \le x_{it} \quad 
    \forall i \in \mathcal{L}, \ j \in \mathcal{J}
\right\}
$$

and $e_i$ is the unit vector with a 1 in the $i$-th position and 0 elsewhere.

### Equivalent Dynamic Program

- Let $y_{ijt} \in \{0,1\}$ indicate whether to accept (1) or reject (0) **resource** $i$ when a **request for product** $j$ arrives (e.g., we allow partially accepting some flight legs when an itinerary uses multiple legs).
- Let $\phi$ be a **fictitious resource** with infinite capacity.
- Let $y_t = \{y_{ijt} : i \in \mathcal{L} \cup \{\phi\}, \ j \in \mathcal{J}\}$.
- Then, $V_t(x_t)$ can be computed as:

$$
\begin{align*}
V_t(x_t) &= \max_{y_t \in \mathcal{Y}(x_t)}
    \left\{ \sum_{j\in \mathcal{J}} p_{jt} 
    \left[ 
        f_j y_{\phi jt} + 
        V_{t+1} \left(x_t - \sum_{i\in \mathcal{L}}y_{ijt}a_{ij}e_i\right) 
    \right] \right\} \tag{DP2} \\
& \text{subject to} \quad y_{ijt} = y_{\phi jt} \quad \forall i \in \mathcal{L}, \ j \in \mathcal{J}
\end{align*}
$$

where

$$
\begin{align*}
\mathcal{Y}_{it}(x_t) &= \left\{ 
    y_{it} \in \{0,1\}^{|\mathcal{J}|} : 
    a_{ij} y_{ijt} \leq x_{it} \quad 
    \forall j \in \mathcal{J} 
\right\} \quad i \in \mathcal{L} \\
\mathcal{Y}_{\phi t}(x_t) &= \left\{ 
    y_{\phi t} \in \{0,1\}^{|\mathcal{J}|} 
\right\} \\
\mathcal{Y}(x_t) &= \mathcal{Y}_{\phi t}(x_t) \times \prod_{i \in \mathcal{L}} \mathcal{Y}_{it}(x_t) \quad \text{(Cartesian product)}
\end{align*}
$$

### Lagrangian Relaxation

- Let $\lambda = \{\lambda_{ijt} : i \in \mathcal{L}, \ j \in \mathcal{J}, \ t \in \mathcal{T}\}$ denote the Lagrangian multiplier.
- The Lagrangian relaxation $V^{\lambda}_t(x_t)$ is defined as:

$$
V^{\lambda}_t(x_t) = \max_{y_t \in \mathcal{Y}(x_t)} 
    \left\{ \sum_{j\in \mathcal{J}} p_{jt} 
    \left[ 
        f_j y_{\phi jt} +
        \sum_{i \in \mathcal{L}} \lambda_{ijt} (y_{ijt} - y_{\phi jt}) + 
        V_{t+1} \left(x_t - \sum_{i\in \mathcal{L}}y_{ijt}a_{ij}e_i\right) 
    \right] \right\}
\tag{LR}
$$

### The Lagrangian Relaxation Algorithm

The Lagrangian relaxation algorithm aims to find an optimal multiplier $\lambda^{*}$ that solves

$$
\begin{align*}
\min_{\lambda} V^{\lambda}_{1}(c_{1})
\end{align*}
$$

As shown in [Topaloglu 2009], we have

$$
\begin{align*}
V_t(x_t) \leq V^{\lambda}_t(x_t) \quad \forall x_t \in \mathcal{C}^{|\mathcal{L}|}, \ t \in \mathcal{T}
\end{align*}
$$

Therefore, $V^{\lambda^{*}}_{1}(c_{1})$ provides a tight bound to $V_{1}(c_{1})$.

It has been shown in [Topaloglu 2009] that given $\lambda$, the value of $V^{\lambda}_{1}(c_{1})$ can be solved by concentrating on one resource at a time. 
Specifically, if $\{\vartheta^\lambda_{it}(x_{it}): x_{it} \in \mathcal{C}, t \in \mathcal{T}\}$ is a solution to the optimality equation

$$
\begin{align*}
\vartheta^\lambda_{it}(x_{it}) = \max_{y_{it}\in\mathcal{Y}_{it}(x_{it})} \left\{ \sum_{j\in\mathcal{J}} p_{jt} \left[ \lambda_{ijt} y_{ijt} + \vartheta^\lambda_{i,t+1}(x_{it} - a_{ij} y_{ijt}) \right] \right\}
\end{align*}
$$

for all $i \in \mathcal{L}$, then

$$
\begin{align*}
V^\lambda_t(x_t) = \sum_{t' = t}^\tau \sum_{j \in \mathcal{J}} p_{jt'} \left[ f_j - \sum_{i \in \mathcal{L}} \lambda_{ijt'} \right]^+ + \sum_{i \in \mathcal{L}} \vartheta^\lambda_{it}(x_{it})
\end{align*}
$$

where $[z]^+ = \max\{z, 0\}$.

It has also been shown in [Topaloglu 2009] that the Lagrangian relaxation $V^{\lambda}_{1}(c_{1})$ is convex in $\lambda$. 
Although we cannot find the gradient of $V^{\lambda}_{1}(c_{1})$ analytically, we can use classical **subgradient methods** to find the optimal multiplier $\lambda^*$.

<br>



<!--
The implementation section should address:
-->

## Our Implementation

Huseyin only remarked in [Topaloglu 2009] that (a) the convex optimization over λ can be handled via classical sub-gradient methods, and (b) the single-resource DP is straightforward to solve. 
He did not present a detailed algorithm or pseudocode. 
Thus, we need to make our own implementation.