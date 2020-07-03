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
<img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;\min_{x}&space;\quad&space;&&space;t&space;(\frac{1}{2}x^\top&space;A&space;x&space;-&space;b^\top&space;x)&space;\&space;-&space;\&space;\sum_{i=1}^{n}&space;\log&space;(q_i&space;-&space;P_{i}x)\\&space;\end{aligned}" title="\begin{aligned} \min_{x} \quad & t (\frac{1}{2}x^\top A x - b^\top x) \ - \ \sum_{i=1}^{n} \log (q_i - P_{i}x)\\ \end{aligned}" /></p>

The barrier method solves a sequence of above problems for increasing positive t, until the duality gap gets very small. You can find the implementation and various experiments in [the barrier method notebook](Barrier%20Method.ipynb).

Moreover, I provide the implementation of the [primal-dual interior-point method](Primal-Dual%20Interior-Point.ipynb). Like the barrier method, primal-dual interior-point techniques aim to compute approximately points on the central path, and its iterations are not necessarily feasible, but they are often more efficient. The Primal-dual interior-point method directly applies Newton root-finding update for solving the perturbed KKT conditions.

#### Ridge Classification With Proximal Gradient Descent

Proximal gradient descent is a generalized form of projection methods used to solve non-differentiable convex optimization problems. In this method, we split non-differentiable convex functions to a single smooth function with a Lipschitz-continuous gradient and a single non-smooth penalty function. Therefore, we write the problem as follows:
<p align = "center">
<img src="https://latex.codecogs.com/svg.latex?\min_{x}&space;f(x)&space;=\&space;\min_{x}&space;\underbrace{g(x)}_{smooth}&space;&plus;&space;\underbrace{h(x)}_{non-smooth}" title="\min_{x} f(x) =\ \min_{x} \underbrace{g(x)}_{smooth} + \underbrace{h(x)}_{non-smooth}" />
</p>
The update step in proximal gradient descent is
<p align = "center">
   <img src="https://latex.codecogs.com/svg.latex?x^&plus;&space;=&space;\underset{z}{\arg\min}\&space;\frac{1}{2t}&space;||z&space;-&space;(x&space;-&space;t\nabla&space;g(x))||_2^2&space;&plus;&space;h(z)" title="x^+ = \underset{z}{\arg\min}\ \frac{1}{2t} ||z - (x - t\nabla g(x))||_2^2 + h(z)" />
</p>

Now consider the binary classification problem with smoothing penalty:
<p align = "center">
   <img src="https://latex.codecogs.com/svg.latex?w^*&space;=&space;\underset{w}{\arg\min}\&space;g(w)&space;&plus;&space;\lambda&space;h(w)&space;=&space;\sum_{i=1}^{n}&space;\log&space;\frac{1&plus;e^{X_{i}&space;w}}{e^{y_i&space;X_{i}&space;w}}&space;&plus;&space;\lambda&space;||w||_{2,c}," title="w^* = \underset{w}{\arg\min}\ g(w) + \lambda h(w) = \sum_{i=1}^{n} \log \frac{1+e^{X_{i} w}}{e^{y_i X_{i} w}} + \lambda ||w||_{2,c}," /></p>
where w is a weight vector as follows:


<p align = "center">
   <img src="https://latex.codecogs.com/svg.latex?w&space;=&space;(w_{(1)},&space;w_{(2)},&space;...,&space;w_{(K)})&space;\in&space;\mathbb{R}^{c_1}&space;\times&space;\mathbb{R}^{c_2}&space;\times&space;...&space;\times&space;\mathbb{R}^{c_K}&space;=&space;\mathbb{R}^{m}" title="w = (w_{(1)}, w_{(2)}, ..., w_{(K)}) \in \mathbb{R}^{c_1} \times \mathbb{R}^{c_2} \times ... \times \mathbb{R}^{c_K} = \mathbb{R}^{m}" /></p>
   
You can examine the codes and analyses of this optimization problem in [this notebook](Proximal%20Gradient%20Descent.ipynb).

## Environment Requirement
* Python 3.6.5

    * NumPy 1.14.3
    * CVXPY 1.1.0
    
## Resources

**Class**:
- [Convex Optimization (Sharif, Spring 2020)](https://inl-lab.net/convex-optimization)

**Textbook**:
- [Convex Optimization (2004)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
