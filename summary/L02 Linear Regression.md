## Summary
1. [[L02 Linear Regression#^77b385|goal of curve fitting]] --> regression problem(mapping problem)
2. Evaluation indicators for *predicted model*:
	1. [[L02 Linear Regression#^69d60a|Expected risk and Empirical risk]]
		1. why can't we directly determine expected risk
		2. what's the difference between expected risk and empirical risk
		3. [[MSE]]
	2. [[MLE]]
		1. what is it and why we use it
		2. the relation between MSE and MLE 
	4. [[L02 Linear Regression#^5e51e2|ERM]]
3. [[L02 Linear Regression#^d146bd|simple linear regression]]
	1. [[Gradient Decent]]
4. [[L02 Linear Regression#^2745fa|when ERM fails]]
	1. [[L02 Linear Regression#^d05b05|condition]]
5. [[L02 Linear Regression#^cb9590|optimal model parameter]]
	1. ==check sufficient condition!!!!!!== --> second deviation > 0 = convex
	2. ==how to get $w_1$==

## Exercise
1. The files data1-1.csv, data1-2.csv and data1-3.csv contain (possibly noisy, incomplete and erroneous) observations $\left(x_i, y_i\right)$. For each of the three datasets, find a parametrized model $f_\theta$ that may have generated the data, i.e. a model $f_\theta$ that reflects the functional relationship $y_i \approx f_\theta\left(x_i\right)$ for all $i$. The model parameters $\theta$ should be unspecified variables here. Their values are specified in Exercise 1.3. What could be the distribution for the $x$-values in each dataset (consider a histogram plot, see matplotlib. pyplot. hist).
	1. histogram can provide us a outlook of $x$ distribution
	2. normalize the data
	3. $w_0, w_1$ with initial value, but we can use curve_fit in sk learn for whole process
	4. define loss function based on MSE
	5. start with initial learning rate or list
	6. based on MLE to find the best model parameters
### Content
1. Problem and motivation of curve fitting ^77b385
	1. Goal: Estimate unknow function $f$ as good as possible
		1. --> $f$ is for mapping problem in math. 
			1. Why estimating $f$
				--> new input examples are easy but output is hard to obtain
				--> predict output by hypothesis $y^{\prime}$ derived from training set
	1. ==to find the best model==

1. Empirical risk minimization
	1. **Expected Risk**: (for estimate unknown $f$)
	2. Let $P(x, y)$ be the probability distribution on the input-output space $\mathcal{Z}=\mathbb{R}^n \times \mathbb{R}$, let $\ell(\hat{y}, y)$ be a loss function, and let $\mathcal{H}=\left\{h: \mathbb{R}^n \rightarrow \mathbb{R}\right\}$ be a hypothesis space. Then
	$$
	E[h]=\int_{\mathcal{Z}} \ell(h(x), y) d P(x, y) .
	$$
	--> Goal: $$
	\text { Find hypothesis } h \in \mathcal{H} \text { that minimizes the expected risk } E[h]
	$$is the expected risk of hypothesis $h \in \mathcal{H}$. 
	With this goal, we can have a standard to judge the good $f$
	1. ==Empirical Risk Minimization Principle==
		1. Determine hypothsis space $\mathcal{H}=\left\{h: \mathbb{R}^n \rightarrow \mathbb{R}\right\}$
		2. Determine [[loss function]] $\ell(h(x), y)$
		3. Minimize ==cost function(empirical risk)== over all hypotheses $h \in \mathcal{H}$. $$
E_m[h]=\frac{1}{m} \sum_{i=1}^m \ell\left(h\left(x_i\right), y_i\right)
$$
		4. Use minimizer $h_m^*$ of $E_m[h]$ as estimate of unknown function $f$
	1. ===[[MSE]] and [[MLE]]==
		1. MSE is a evaluation factor to evaluate the model(is it good)
		2. MLE is an approach to estimate the best model parameters based on observed data(it belongs to stastical inference)
	2. Empirical Risk Minimization
		1. $E_m[h]$ converges probabilistically to $\mathrm{E}[h]$ for fixed $h$ with increasing $m$ 
		2. $E_m\left[h_m^*\right]$ converges probabilistically to $\mathrm{E}\left[h_{\infty}^*\right]=\mathrm{E}\left[h^*\right]$ with increasing $m$
		==ERM only works when we restrict the hypothesis space== ^69d60a ^5e51e2 ^d05b05
1. Simple linear regression(n = 1) ^d146bd
	1. Function: $y = w_0 +w_1 x$
		1. $w_0$: Bias, determine position along y-axis
		2. $w_1$: weight, determine the slope
	2. Loss
		1. Squared loss: focus on one data point
		2. MSE: focus on the whole dataset
	3. ==Optimal Parameter==
2. Gradient descent for SLR[[Gradient Decent]]
	1. Goal: to find the best weight, start with random weights
	2. ==Attention==: calculate the step(deviation for every parameter first)
	3. Choice of learning rate
	4. Repeat until termination:
		1. termination after maximum number of simultaneous steps
		2. terminate if terminate if $\left\|\nabla J\left(w_0, w_1\right)\right\|<\varepsilon$
3. when ERM fails
	$$f(x)= \begin{cases}0, & 0 \leq x<1 / 2 \\ 1, & \frac{1}{2} \leq x \leq 1\end{cases}$$$\left(x_1, y_1\right), \ldots,\left(x_m, y_m\right) \in[0,1] \times[0,1]$ training set with $y_i=f\left(x_i\right)$ for all $i$ (no noise assumed)$$h_m(x)= \begin{cases}y_i, &  x = x_i \\ 1, &else\end{cases}$$ $h_m$ memorizes training set and takes value 1 elsewhere ^2745fa
	- Expected error = 1/2
	- Empirical error = 0
4. optimal model parameters
	Write $\partial\left(w_0, w_n\right):=E_m\left[h_w\right]=\frac{1}{2 m} \sum_{i=1}^m\left(w_0+w_n x_i-y_i\right)^2$
	(0) $\frac{\partial f}{\partial w_0}\left(w_0, w_1\right)=\frac{1}{2 m} \sum_{i=1}^m 2\left(w_0+w_1 x_i-y_i\right)$
	$$=w_0+w_1 \underbrace{\frac{1}{m} \sum_{i=1}^m x_i}_{=\bar{x}}\underbrace{\frac{1}{m} \sum_{i=1}^m y_i}_{=\bar{y}}$$ ^cb9590
	- Set partial derivative to zero ad solve for wo (necessary condition of local min.)
	$$\begin{aligned}& \frac{\partial f}{\partial_{\omega_0}}\left(\omega_0, \omega_1\right)=\omega_0+\omega_1 \bar{x}-\bar{y}=0 \\& \Rightarrow \omega_0^*=\bar{y}-\omega_1 \bar{x}\end{aligned}$$
	- check sufficient condition for local min.
	$$\frac{\partial^2 f}{\partial w_0}\left(w_0, w_n\right)=1>0 \Rightarrow l_{0 \mathrm{cal}} \mathrm{min}$$
	(2)$$\begin{aligned}& \frac{\partial \gamma}{\partial w_1}\left(w_0, w_1\right)=\frac{1}{2 m} \sum_{i=1}^m 2\left(w_0+w_1 x_i-x_i\right) \cdot x_i \\& =\frac{1}{m} \sum_{i=1}^m\left(\bar{y}-w_1 \bar{x}+w_1 x_i-y_i\right) x_i \\& =\frac{1}{m} \sum_{i=1}^m\left(w_n\left(x_i-\bar{x}\right)-\left(y_i-\bar{y}\right)\right) x_i \\& =\frac{1}{m}\left(w_1 \sum x_i\left(x_i-\bar{x}\right)-\sum_{i=1}^m x_i\left(y_i-\bar{y}\right)\right) \stackrel{!}{=} 0 \\& \Leftrightarrow \quad w_1 \sum x_i\left(x_i-\bar{x}\right)=\sum x_i\left(y_i-\bar{y}\right) \\& \Leftrightarrow \quad w_n=\frac{\sum x_i\left(y_i-\bar{y}\right)}{\sum x_i\left(x_i-\bar{x}\right)} \\& =\frac{\sum\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)}{\sum\left(x_i-\bar{x}\right)^2} \\&\end{aligned}$$$\frac{1}{m} \sum\left(x_i+c\right)\left(y_i-\bar{y}\right)=\frac{1}{m} \sum x_i\left(y_i-\bar{y}\right)+\underbrace{c\left(\frac{1}{m} \sum y_i-\bar{y}\right)}_{=0}$
	Save for denowivater ad $\frac{1}{m}$ cancels out.
	Second derivative:$$\frac{\partial^2 f}{\partial^2 w_1}\left(w_0, w_n\right)=\sum_i x_i^2 \geq 0 \Rightarrow l_0 \text { cal min if } \sum x_i^2 \neq 0$$In summary, ( $\left.W_0 w_1\right)$ as specified is a global min if $\sum x_i^2 \neq 0$

