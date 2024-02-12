## Summary
1. [[L05 Bias-Variance Decomposition#^cfcbe1|bias and variance estimator]]
	1. corrected sample variance and why
2. definition of learner
3. why does $h^*$ need not be in H --> learn more feature and avoid overfitting
4. [[L05 Bias-Variance Decomposition^a9d999|expected generalization error decomposition]]
5. [[L05 Bias-Variance Decomposition^607c2d|overfitting and underfitting]]

## Exercise
1. MSE in dependence of polynomial order: increase the polynomial order, the MSE decreases in training stage
	1. too small --> underfitting, MSE large(bias)
	2. too large --> overfitting, MSE large (varaince)
2. generalization error in dependence of training examples
3. MSE and cross valiadation error in dependence of polynomial order
4. Let $x_1, x_2, \ldots, x_m$ be independent and identically distributed random variables with probability distribution $P$ having mean $\mu$ and variance $\sigma^2$. Compute the variance of the sample mean $\bar{x}=\frac{1}{n} \sum_{i=1}^n x_i$ when it is considered as an estimator of the population mean $\mu$.
5. why there's a tradeoff between bias and variance
### Content
1. Bias and variance of estimators
	Let $x_1, x_2, \ldots, x_m \sim_{\text {iid. }} P, \mathbb{E}\left[x_1\right]=\mu, \mathbb{V}\left[x_1\right]=\sigma^2$

	The sample mean $\bar{x}=\frac{1}{m} \sum_{i=1}^m x_i$ is unbiased
	The variance of the sample mean is $\mathbb{V}[\bar{x}]=\frac{1}{m} \sigma^2$
	The sample variance $\frac{1}{m} \sum_{i=1}^m\left(x_i-z\right)^2$ is biased for $\mathrm{z}=\bar{x}$ and unbiased for $\mathrm{z}=\mu$.
	The corrected sample variance $\frac{1}{m-1} \sum_{i=1}^m\left(x_i-\bar{x}\right)^2$ is unbiased
 ^cfcbe1
2. Bias-variance decomposition
	1. Motivation: Bias-variance decomposition explains the phenomena of under- and overfitting.
		--> Bias-variance decomposition explains propety of "U-shape" test error as a function of model complexity
	2. Generalization error of hypothesis $h_{\mathcal{D}} \in \mathcal{H}$: $$
\operatorname{Err}\left[h_{\mathcal{D}}\right]=\mathbb{E}_{x, \varepsilon}\left[\left(h_{\mathcal{D}}(x)-y(x, \varepsilon)\right)^2\right]
$$
	3. Expected generalization error of Learner L: $$
\operatorname{Err}[L]=\mathbb{E}_{\mathcal{D}}\left[\mathbb{E}_{x, \varepsilon}\left[\left(h_{\mathcal{D}}(x)-y(x, \varepsilon)\right)^2\right]\right]
$$
	> Hypothesis $h_D$ picked by learner and generalization error of $h_D$ depend on dataset $D$.
	
==The difference of genelirization error of hypothesis and learner: learner consider fine-tune on hyper parameters and multiple hypothesis.==
	4. Expected generalization error of $L$ in uncluttered notation$$\operatorname{Err}[L]=\mathbb{E}\left[\left(h_{\mathcal{D}}(x)-y\right)^2\right]$$
	5. Expected hypothesis of Learner $L$ at $x$: $$h^*(x)=\mathbb{E}_{\mathcal{D}}\left[h_{\mathcal{D}}(x)\right]$$
	==$h^*$ need not be in $\mathcal{H}$ == -->learn more features and avoid overfitting

 3. Decomposition
 $$
\mathbb{E}_{\mathcal{D}, x, \varepsilon}\left[\left(h_{\mathcal{D}}(x)-y\right)^2\right]=\underbrace{\mathbb{E}_x\left[\left(f(x)-h^*(x)\right)^2\right]}_{\text {bias }}+\underbrace{\mathbb{E}_{\boldsymbol{x}, \mathcal{D}}\left[\left(h_D(x)-h^*(x)\right)^2\right]+}_{\text {variance }}+\underbrace{\sigma^2}_{\text {noise }}
$$
	Bias: error caused by simplifying assumptions built into the learner $L=(\mathcal H, A)$
	variance: error caused by flexibility of learner
	Noise: irreducible error ^a9d999

$\mathcal{D}_1, \ldots, \mathcal{D}_L=$ training sets of size $m$
$\mathcal{D}_{\text {test }}=\left(\left(x_1, y_1\right), \ldots,\left(x_p, y_p\right)\right)$ test set of size $p$
$$
\begin{array}{rlr}
\widehat{h}^*(x)=\frac{1}{L} \sum_{\ell=1}^L h_{\mathcal{D}_{\ell}}(x) & & \left(\text { estimates } \mathbb{E}_{\mathcal{D}}\left[h_{\mathcal{D}}(x)\right]\right) \\
\widehat{\mathbb{E}}_{\text {bias }} & =\frac{1}{p} \sum_{i=1}^p\left(f\left(x_i\right)-\widehat{h}^*\left(x_i\right)\right)^2 & \left(\text { estimates } \mathbb{E}_x\left[\left(f(x)-h^*(x)\right)^2\right]\right) \\
\widehat{\mathbb{E}}_{\text {var }} & =\frac{1}{p} \sum_{i=1}^p \frac{1}{L} \sum_{\ell=1}^L\left(h_{\mathcal{D}_{\ell}}\left(x_i\right)-\widehat{h}^*\left(x_i\right)\right)^2 & \left(\text { estimates } \mathbb{E}_{x, \mathcal{D}}\left[\left(h_D(x)-h^*(x)\right)^2\right]\right)
\end{array}
$$


1. High Bias: more training data does not help much to reduce the test error
2. High variance: more training data is likely to reduce the test error
3. [math background](https://zhuanlan.zhihu.com/p/40481534)

==Rule of Thumb advices== ^607c2d
1. High train and high test error
	1. high bias
	2. choose a more flexible hypothesis space
	3. and add more features
	4. reduce regularization parameter
2. low train and high test error
	1. high variance
	2. sample more training examples
	3. reduce number of features
	4. increase regularization parameter


