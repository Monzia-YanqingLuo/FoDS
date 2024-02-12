## Summary
1. [[L09+10 Linear classifier#^1513f2|geometric interpretation]]
2. [[L09+10 Linear classifier#^8fd4ab|fisher consistency]]
3. [[L09+10 Linear classifier#^3fa0c5|logistic regression]]
4. [[L09+10 Linear classifier#^069e0d|probabilistic interpretation]]
	1. discriminative classifier
	2. generative classifier
5. softmax
## Exercise
1. **loss gradient in matrix notation**
	Let $\left(x_1, y_1\right), \ldots,\left(x_m, y_m\right) \in \mathbb{R}^{n+1} \times\{0,1\}$ be a training set with augmented inputs $x_i=\left(1, x_{i 1}, \ldots, x_{i n}\right)$. Consider the $L_2$-regularized loss function$$J(w)=-\frac{1}{m} \sum_{i=1}^m y_i \log \left(\sigma\left(w^T x_i\right)\right)+\left(1-y_i\right) \log \left(1-\sigma\left(w^T x_i\right)\right)+\frac{\lambda}{2 m} \sum_{j=1}^n w_j^2$$Derive the loss function and its gradient in matrix notation.
2. Show that the logistic loss and the squared error loss are both Fisher consistent.
3. **Softmax**
	The goal of this exercise is to derive the softmax function$$h_k(x)=P(y=k \mid x)=\frac{\exp \left(w_k^{\top} x\right)}{\sum_{l=1}^c \exp \left(w_l^{\top} x\right)} \quad, k=1, \ldots, c$$by modeling the Posterior with a categorical distribution. For this, complete the sketch of the probabilistic interpretation of softmax regression. In particular,
	a) Show that
	- $p_k(x)=\frac{\exp \left(w_k^T x\right)}{1+\sum_{l=1}^{c-1} \exp \left(w_l^T x\right)}, k=1, \ldots, c-1$,
	- $p_c(x)=\frac{1}{1+\sum_{l=1}^{c-1} \exp \left(w_l^T x\right)}$
	 b) Find $w_1^{\prime}, \ldots, w_c^{\prime}$ such that $p_k(x)=\frac{\exp \left(w_k^T x\right)}{\sum_{l=1}^c \exp \left(w_l^T x\right)}$, for all $k=1, \ldots, c$ and all $x \in \mathbb{R}^{n+1}$.
	 c) Present the whole probabilistic derivation.
### Content
1. Geometric interpretation ^1513f2
	1. Linear function$$h(x)=w_0+w_1 x_1+w_2 x_2$$
	2. Decision boundary$$\mathcal{B}_h=\left\{x \in \mathbb{R}^2: h(x)=0\right\}$$
	3. $\mathcal{B}_h$ is a hyperplane (if $w \neq 0$ )
	4. $w$ is orthogonal to $\mathcal{B}_h$
	5. $w$ points to region $\mathcal{R}_1$
	6. signed distance of 0 from $\mathcal{B}_h$ is$$r_0=d\left(0, \mathcal{B}_h\right)=\frac{w_0}{\|w\|}$$
	7. signed distance of $\mathrm{x}$ from $\mathcal{B}_h$ is$$r=d\left(x, \mathcal{B}_h\right)=\frac{h(x)}{\|w\|}$$
2. Goal: minimize the expected risk --> Approach: minimize empirical risk
3. Loss
	1. zero-one loss
		1. Notation: $$\ell(\hat{y}, y)=\left\{\begin{array}{lll}1 & : & y \hat{y}<0 \\0 & : & y \hat{y} \geq 0\end{array}\right.$$
		2. -->$l(\alpha), \quad \alpha=y \hat{y}$
		3. Empirical risk $E_m[h]$ is 
			1. a step function
			2. discontinuous
			3. non-convex
		4. Implications:
			1. minimizing $E_m[h]$ is NP-hard
			2. gradient descent fails
		5. $\Longrightarrow$ ==surrogate loss functions==
		6. the advantage of logistic loss
	2. Fisher consistency: Let $\ell$ be the zero-one loss and $\phi$ a surrogate loss function. Consider the Bayes risks $E_{\ell}^*=\min E_{\ell}[h]$ and $E_\phi^*=\min E_\phi[h]$. The surrogate loss $\phi$ is Fisher consistent if for any sequence $\left(h_n\right)$ of functions:$$E_\phi\left[h_n\right] \rightarrow E_\phi^* \text { implies } E_{\ell}\left[h_n\right] \rightarrow E_{\ell}^*$$ ^8fd4ab
		1. $\text { If } \phi \text { (as a function of } \alpha=y \hat{y} \text { ) is convex, differentiable at } 0 \text { with } \phi^{\prime}(0)<0 \text {, then } \phi \text { is Fisher consistent. }$
		2. 为什么fisher consistency是小于0，而不是等于0
	3. Empirical risk minimization
		1. general:
			1. General loss function： $\ell(\hat{y}, y)$
			2. Empirical risk： $J(w)=\sum_{i=1}^m \ell\left(h_w\left(x_i\right), y_i\right)$
			3. Gradient descent：$w_j \leftarrow w_j-\eta \frac{\partial}{\partial w_j} J(w) \quad \text { for all } j$
		2. Linear regression: 
			1. squared loss function: $\phi(\hat{y}, y)=(\hat{y}-y)^2$
			2. Empirical Risk: $J(w)=\frac{1}{2 m} \sum_{i=1}^m\left(h_w\left(x_i\right)-y_i\right)^2$
			3. Gradient descent: $w_j \leftarrow w_j-\frac{\eta}{m} \sum_{i=1}^m\left(h_w\left(x_i\right)-y_i\right) x_{i j} \quad$ for all $j$
		3. Logistic regression:
			1. logistic loss function: $\phi(\hat{y}, y)=\log (1+\exp (-y \hat{y}))$
			2. empirical risk: $J(w)=\frac{1}{m} \sum_{i=1}^m \log \left(1+\exp \left(-y_i \cdot h_w\left(x_i\right)\right)\right)$
			3. gradient descent: $w_j \leftarrow w_j+\frac{\eta}{m} \sum_{i=1}^m \frac{y_i \cdot x_{i j}}{1+\exp \left(y_i h_w\left(x_i\right)\right)} \quad$ for all $j$
		4. geometric visualization of update rule(single training example):
			1. $w^{\prime}=w+\alpha \frac{y x}{1+e^{y h(x)}}$
4. Probabilistic interpretation ^069e0d
	1. discriminative classifier:
		1. model posterior $p(y|x)$
		2. example: linear classifier, logistic regression
		3. assume functional form of discriminant e.g. linear
		4. estimate weights of hypothesis space e.g. maximum likelihood
	2. generative classifier:
		1. estimate joint distribution: $p(x,y) =p(x|y)p(y)$
		2. example: naive bayes
		3. make explicit assumptions on distribution e.g. Gaussian, multinomial,...
		4. estimate parameters of distribution e.g. maximum likelihood
	3. Logistic regression
		1. Posterior $P(\mathrm{y} \mid \mathrm{x})$ follows Bernoulli distribution
			- probability of $y=1$ given $x: p_x=P(y=1 \mid x)$
			- probability of $y=0$ given $x$ : $q_x=P(y=0 \mid x)=1-p_x$
		2. Bayes decision rule: https://cdn.mathpix.com/snip/images/VTT5nOdRkpKXxKtaw9eDHecPA46UeEQp9PPUqZ9LK_s.original.fullsize.png
		3. Goal: estimate unknown function: $f: \mathcal{X} \rightarrow[0,1], \quad x \mapsto f(x)=p_x$
		4. equicalent decision rule: https://cdn.mathpix.com/snip/images/AxVgqoQYJV1O5836HaJ2NzjI1_fedhg-cWsrCwhrEgg.original.fullsize.png
		5. equivalent decision boundaries: $\mathcal{B}_f:=\left\{x \in \mathcal{X}: p_x=1-p_x\right\}=\{x \in \mathcal{X}: g(x)=0\}=: \mathcal{B}_g$
		6. Logistic regression: models $g(x)=\log \left(p_x / 1-p_x\right)=w^T x=h_w(x)$ as a linear function ^3fa0c5
		7. relationship between function $f(x)=p_x$ and $g(x)$
			1. logit: $g(x)=\log \frac{f(x)}{1-f(x)}$
			2. logistic/sigmoid: $f(x)=\sigma(g(x))$, where $\sigma(z)=\frac{1}{1+\exp (-z)}=\frac{\exp (z)}{1+\exp (z)}$
				1. properties:
					1. symmetry: $\sigma(z)+\sigma(-z)=1$
					2. derivative: $\sigma^{\prime}(z)=\sigma(z)(1-\sigma(z))$
		8. Logistic regression model: $f(x)=\sigma\left(w^{\top} x\right)=\frac{1}{1+\exp \left(-w^{\top} x\right)}$
		9. Loss:
			1. empirical risk arises from MLE: maximes likelihood = minimize negative log-likelihood:$-\log (L(w))=-\sum_{i=1}^m y_i \log \left(\sigma\left(w^{\top} x_i\right)\right)+\left(1-y_i\right) \log \left(1-\sigma\left(w^{\top} x_i\right)\right)$
			2. Empirical risk: $J(w)=-\frac{1}{m} \sum_{i=1}^m y_i \log \left(\sigma\left(w^{\top} x_i\right)\right)+\left(1-y_i\right) \log \left(1-\sigma\left(w^{\top} x_i\right)\right)$
			3. gradient decent: $w_j \leftarrow w_j-\frac{\eta}{m} \sum_{i=1}^m\left(\sigma\left(w^{\top} x_i\right)-y_i\right) x_{i j} \quad$ for all $j$
			4. Empirical risk with L2-regularization in matrix form:$$J(w, \lambda)=-\frac{1}{m}\left(y^{\top} \log (h)+(1-y)^{\top} \log (1-h)\right)+\frac{\lambda}{2 m} w^{\top} \widetilde{I} w$$
			5. Gradient desent: $w \leftarrow w-\frac{\eta}{m}\left(X^{\top}(h-y)+\lambda \widetilde{I} w\right)$