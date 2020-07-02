# Convex Optimization
This repository provides implemented algorithms for several convex optimization problems. All code is written in Python 3, using NumPy and CVXPY. Jupyter notebooks are provided to show analyses.

**Note**: These are implemented algorithms and study for selected homeworks of [convex optimization course](https://inl-lab.net/convex-optimization) taught by Prof. Jafari Siavoshani in Spring 2020.

## List of Notebooks:

#### Inequality-Constrained Quadratic Optimization with Second Order Methods
Consider quadratic problem with inequality Constraints:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;\min_{x}&space;\quad&space;&&space;\frac{1}{2}x^\top&space;A&space;x&space;-&space;b^\top&space;x\\&space;\textrm{s.t.}&space;\quad&space;&&space;Px&space;\preccurlyeq&space;q,\\&space;\end{aligned}" title="\begin{aligned} \min_{x} \quad & \frac{1}{2}x^\top A x - b^\top x\\ \textrm{s.t.} \quad & Px \preccurlyeq q,\\ \end{aligned}" /></p>

where P and q are random matrix and vector respectively. We can write the barrier problem as
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;\min_{x}&space;\quad&space;&&space;t&space;(\frac{1}{2}x^\top&space;A&space;x&space;-&space;b^\top&space;x)&space;\&space;-&space;\&space;\sum_{i=1}^{n}&space;\log&space;(q_i&space;-&space;P_{i,*}x)\\&space;\end{aligned}" title="\begin{aligned} \min_{x} \quad & t (\frac{1}{2}x^\top A x - b^\top x) \ - \ \sum_{i=1}^{n} \log (q_i - P_{i,*}x)\\ \end{aligned}" /></p>

## Environment Requirement
* Python 3.6.5

    * NumPy 1.14.3
    * CVXPY 1.1.0
    
## Resources

**Class**:
- [Convex Optimization (Sharif, Spring 2020)](https://inl-lab.net/convex-optimization)

**Textbook**:
- [Convex Optimization (2004)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
