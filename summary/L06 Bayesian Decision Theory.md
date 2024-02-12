## Summary
1. [[L06 Bayesian Decision Theory#^6785e6|bayes theorem]]
2. [[L06 Bayesian Decision Theory#^6cc657|bayes decision theory]]
	1. posterior
	2. prior
	3. likeli
	4. decision boundary
	5. [[L06 Bayesian Decision Theory#^4acffb|loss]]
		1. zero-one loss
		2. non-symmetric loss
		3. squared loss
		4. expected condition risk(loss)
		5. expected loss
	6. [[L06 Bayesian Decision Theory#^0c005f|Bayes classifier]]
3. [[L06 Bayesian Decision Theory#^890448|discriminant functions]]
	1. [[L06 Bayesian Decision Theory#^bebcc7|decision rule]]
	2. [[L06 Bayesian Decision Theory#^8fb66f|geometric properties]]
	3. generalize bayes decision rule
	4. define class regions and decision boundaries
4. decision rule with zero-one loss:
	1. [[L06 Bayesian Decision Theory#^7a92ab|bayes]]
	2. [[L06 Bayesian Decision Theory#^29abe4|bayes with discriminant]]
## Exercise
1. **Determine Bayes Decision Rule**
	For a two-class classification problem with class labels 0 and 1 and posteriors $P(y=0 \mid x), P(y=1 \mid x)$ consider the loss function $l(0,1)=2, l(1,0)=1, l(0,0)=l(1,1)=0$. Determine the Bayes decision rule for input variable $x$.
	```markdown
	1. calculate the expected conditional loss
	2. according to the decision rule based on the error
```
2. **Beneulli with Gaussian**
	Consider the following two-class classification problem: Class 0 occurs with probability $p$, class 1 occurs with probability $1-p$. For $i=0,1$ let $x_i \sim \mathcal{N}\left(\mu_i, \sigma_i^2\right)$ be a one-dimensional, normally distributed random variable with class label $i$. The Bayes classifier with zero-one loss assigns to an $x \in \mathbb{R}$ the class $i$ with the maximum posterior probability $P(y=i \mid x)$.
	a) Show that the decision boundary of the Bayes classifier can be described by the zero set of a polynomial of maximum degree 2 .
	b) In which case is the degree 1? Specify the decision rule for this case. What is the influence of the priors here?
	c) In which case is the degree 0? Specify the decision rule for this case. What is the influence of the priors here?
	```markdown
	1. according to the decision boundary based on posterior
	2. simplify the polynomial
```
3. **Decision Boundary**
	The d-dimensional normal distribution with mean $\mu \in \mathbb{R}^d$ and covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$ has the probability density$$p(x)=\frac{1}{(2 \pi)^{d / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right) .$$For $i=0,1$ let $x_i \sim \mathrm{N}\left(\mu_i, \Sigma_i\right)$ be d-dimensional, normally distributed random variables with class label $i$. Consider the Bayes classifier with equal priors for both classes. Show that $\Sigma_0=\Sigma_1, \mu_0 \neq \mu_1$ implies a linear decision boundary. Describe the decision boundary for $\Sigma_0=\Sigma_1, \mu_0=\mu_1$.
	```markdown
	1. still two classes
	2. decision boundary calculation
	3. simplify the decision boundary
```
### Content
1. Backgroud
	1. decision rule: a function that maps an observation $x$ to an approapriate class label $y$
	2. Prior probability: reflects our belief that class $y$ occurs before data $x$ is observed
	3. class-conditional probability density $p(x \mid y)$: probability density to observe $x$ given the class is $y$.
	4. Bayes formula: [[Bayes Theroem]]
	5. Bayes decision rule/ Bayes classifier:$$
\begin{aligned}
& y=\left\{\begin{array}{lll}
0 & : & P\left(y_0 \mid x\right)>P\left(y_1 \mid x\right) \\
1 & : & \text { otherwise }
\end{array}\right. \\
& y=\left\{\begin{array}{lll}
0 & : & p\left(x \mid y_0\right) P\left(y_0\right)>p\left(x \mid y_1\right) P\left(y_1\right) \\
1 & : & \text { otherwise }
\end{array}\right.
\end{aligned}
$$ ^6785e6

Bayes decision rule when priors $P(y)$ are equal
$$
y=\left\{\begin{array}{lll}
0 & : & p\left(x \mid y_0\right)>p\left(x \mid y_1\right) \\
1 & : & \text { otherwise }
\end{array}\right.
$$

2. bayes decision rule ^6cc657
	1. probability of error: $$P(\text { error } \mid x)=\min \left\{P\left(y_0 \mid x\right), P\left(y_1 \mid x\right)\right\}$$
	2. **minimize probability of error**
	3. loss function: cost incurred by predicting $\hat{y}$ when true class is $y$, prediction$\hat{y} \in \mathbb{R}$, class label $y \in\{1, \ldots, c\}$ ^4acffb
		1. zero-one loss: misclassifications have unit loss$$\ell(\widehat{y}, y)=\left\{\begin{array}{lll}1 & : & \widehat{y} \neq y \\0 & : & \widehat{y}=y\end{array}\right.$$
		2. non-symmetric loss: $$\begin{aligned}& \ell(\widehat{\text { ham }}, \text { spam })<\ell(\widehat{\text { spam }}, \text { ham }) \\& \ell(\text { malignant, benign })<\ell(\text { benign, malignant })\end{aligned}$$ 这里考察分类错误成本，选择错误成本低的
		3. squared loss: $$\ell(\hat{y}, y)=(\hat{y}-y)^2$$ ==为什么和regression里面的squared loss 不一样==
		4. Expected condition risk(loss) of prediction $y_l$:$$R\left(y_l \mid x\right)=\sum_{k=1}^c \ell\left(y_l, y_k\right) P\left(y_k \mid x\right)$$
		5. Expected risk(loss):$$R[h]=\int R(h(x) \mid x) p(x) d x$$ $$\text { where } h: \mathbb{R}^n \rightarrow\{1, \ldots, \mathrm{c}\} \text { is a decision rule }$$
		6. Decision RULE: $h$ assign class $h(x)=y$ to input $x$
	4. Goal: find decision ruhe $h(x)$ that minimize the expected risk
	5. Bayes decision Rule/ Bayes classifier:$$f(x) \in \arg \min _{l=1}^c R\left(y_l \mid x\right)=\arg \min _{l=1}^c \sum_{k=1}^c \ell\left(y_l, y_k\right) P\left(y_k \mid x\right)$$ ^0c005f
	6. bayes risk$R[f]$: 
		1. $R[f]=\min _h R[h], \text { where } f \text { is the true function }$
		2. best performance that can be achieved
	7. bayes decision rule for zero-one loss: ^7a92ab
		1. Posterior: $f(x)=\arg \max _{k=1}^c P\left(y_k \mid x\right)$
		2. Likelihood x prior: $f(x)=\arg \max _{k=1}^c p\left(x \mid y_k\right) P\left(y_k\right)$
		3. identical priors: $f(x)=\arg \max _{k=1}^c p\left(x \mid y_k\right)$

3. Discriminant Functions ^890448
	1. decision rule: $$l=\arg \max _{k=1}^c h_k(x)$$ compute score $h_k(x)$ for every class $y_k$; assign $x$ to class $l$ with max score ^bebcc7
		1. decision rule is invariant under monotonous transformations: $$l=\arg \max _{k=1}^c h_k(x)=\arg \max _{k=1}^c g\left(h_k(x)\right)$$
	2. **geometric properties**
		$h_k(x)=-R\left(y_k \mid x\right)=P\left(y_k \mid x\right)$
		2. class regions $\mathcal{R}_k$
			1. discriminants $h_k$ devide the input space $R^n$ into $c$ class regions $\mathcal{R}_1$ ... $\mathcal{R}_c$
			2. $x \in \mathcal{R}_l$ implies $h_l(x) \geq h_k(x)$ for all $k \neq l$    --> 相当于最大化后验概率
		3. decision boundaries $\mathcal{B}_{kl}$
			1. adjacent class regions $\mathcal{R}_k,\mathcal{R}_l$ are seperated by decision boundaries$$\mathcal{B}_{k l}=\left\{x \in \mathbb{R}^n: h_k(x)=h_l(x)\right\}$$ ^8fb66f
	1. single discriminant: $h(x)=h_1(x)-h_0(x)$, $l=\left\{\begin{array}{lll}1 & : & h(x) \geq 0 \\ 0 & : & \text { otherwise }\end{array}\right.$
	2. decision rule for bayes classifier with zero-one loss:  ^29abe4
		1. single discriminant: $h(x)=P\left(y_1 \mid x\right)-P\left(y_0 \mid x\right)$
		2. single tranformed discriminant: $h(x)=\log \frac{p\left(x \mid y_1\right)}{p\left(x \mid y_0\right)}+\log \frac{P\left(y_1\right)}{P\left(y_0\right)}$
			1. where $\log \frac{P\left(y_1\right)}{P\left(y_0\right)}$ is the threshold for decision boundary

^e2f676
