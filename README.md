# Convex Optimization
This repository provides implemented algorithms for several convex optimization problems. All code is written in Python 3, using NumPy and CVXPY. Jupyter notebooks are provided to show analyses.

**Note**: These are implemented algorithms and study for selected homeworks of [convex optimization course](https://inl-lab.net/convex-optimization) taught by Prof. Jafari Siavoshani in Spring 2020.

## Table of Contents:

#### Inequality-Constrained Quadratic Optimization with Second Order Methods
Consider quadratic problem with inequality constraints:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;\min_{x}&space;\quad&space;&&space;\frac{1}{2}x^\top&space;A&space;x&space;-&space;b^\top&space;x\\&space;\textrm{s.t.}&space;\quad&space;&&space;Px&space;\preccurlyeq&space;q,\\&space;\end{aligned}" title="\begin{aligned} \min_{x} \quad & \frac{1}{2}x^\top A x - b^\top x\\ \textrm{s.t.} \quad & Px \preccurlyeq q,\\ \end{aligned}" /></p>

where P and q are random matrix and vector respectively. We can write the barrier problem as
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;\min_{x}&space;\quad&space;&&space;t&space;(\frac{1}{2}x^\top&space;A&space;x&space;-&space;b^\top&space;x)&space;\&space;-&space;\&space;\sum_{i=1}^{n}&space;\log&space;(q_i&space;-&space;P_{i,*}x)\\&space;\end{aligned}" title="\begin{aligned} \min_{x} \quad & t (\frac{1}{2}x^\top A x - b^\top x) \ - \ \sum_{i=1}^{n} \log (q_i - P_{i,*}x)\\ \end{aligned}" /></p>

The barrier method solves a sequence of above problems for increasing positive t, until the duality gap gets very small. You can find the implementation and various experiments in [the barrier method notebook](Barrier%20Method.ipynb).

Moreover, I provide the implementation of the [primal-dual interior-point method](Primal-Dual%20Interior-Point.ipynb). Like the barrier method, primal-dual interior-point techniques aim to compute approximately points on the central path, and its iterations are not necessarily feasible, but they are often more efficient. The Primal-dual interior-point method directly applies Newton root-finding update for solving the perturbed KKT conditions.

## Environment Requirement
* Python 3.6.5

    * NumPy 1.14.3
    * CVXPY 1.1.0
    
## Resources

**Class**:
- [Convex Optimization (Sharif, Spring 2020)](https://inl-lab.net/convex-optimization)

**Textbook**:
- [Convex Optimization (2004)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
