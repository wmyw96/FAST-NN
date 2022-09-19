## Implementation and Numerical Studies of FAST-NN



### EXP 1: Performance of FAR Model

We first use the simulation studies to illustrate the (1) near-oracle performance of our FAR-NN estimator and (2) the necessity of using a predefined fixed diversified projection matrix. 

In particular, we consider the four estimators.

[1] oracle [2] far-nn [3] vanilla [4] joint training

The data generating process is as follows:

- Covariate $\mathbf{x} \in \mathbb{R}^p$ admits a linear factor model with the number of factors $r=5$. The factor loading matrix has i.i.d. $\mathrm{Unif}[-\sqrt{3},\sqrt{3}]$ entries. The latent factor and the idiosyncratic components have i.i.d. $\mathrm{Unif}[-1,1]$ entries.
- The regression function is $m^*(\mathbf{f}) = \sum_{j=1}^r m_j^*(f_j)$, where $m_j^*$ are selected from the functions $\{\cos(\pi x), \sin(x), (1-|x|)^2, 1/(1+e^{-x}), 2\sqrt{|x|}-1\}$ randomly.
- The response variable $y=m^*(\mathbf{f})+\varepsilon$, where $\varepsilon$ is another independent $\mathcal{N}(0,0.3)$ random variable.

We consider the case with fixed $n_{train}=500$, and use another $m=50$ samples to calculate the diversified projection matrix. We allow overestimating the number of covariates slightly, that is, to set the number of diversified weights to be $\overline{r}=10$. 

For other parameters, we simply let them be fixed across different models. To be specific, we let $N=300$, depth $L=4$. We use Adam optimizer with batch size $64$, learning rate $1e-4$. 

To report the test performance, we use the validation set with a sample size $n_{valid} = 150$, and report the test performance of the model with the smallest validation error. The test performance is calculated via
$$
\frac{1}{n_{test}} \sum_{i=1}^{n_{test}} \{\hat{m}(\mathbf{x_i}) - m^*(\mathbf{f}_i)\}^2
$$
using other $n_{test}=10000$ samples. 



We consider the case where $p=100, 200, 500, 800, 1000, 2000, 3000, 4000, 5000, 6000$, and for each case, we repeat it 10 times.

