---
title: 'Variational inference for scaling and merging in crystallography'
date: 2024-07-01
modified: 2024-07-01
permalink: /posts/VI-merging/
color_tag: "study-notes"
tags:
  - ML
  - crystallography
---

[This](https://pmc.ncbi.nlm.nih.gov/articles/PMC10478637/) perspective describes existing algorithms for scaling and merging in crystallography and the recent introduction of variational inference (VI) in this field.

Conventional methods estimate the structure factor amplitude $$F= \sqrt{\bar{I}}$$ by averaging redundant measurements of the reflection intensity $$I$$ weighted by their uncertainty $$\sigma_I$$ after performing a scaling that corrects for systematic errors. This approach can be interpreted as the maximum likelihood estimation (MLE), where using normal distribution as the error model, the likelihood of $$N$$ measurements

$$
    p_\theta(I) = \prod^N_i p_\theta(I_i)
                      = \prod^N_i \frac{1}{\sigma_{I_i}\sqrt{2\pi}}
                       \exp\left[-\frac{(I_i - \theta)^2}{2\sigma^2_{I_i}}\right]
$$

is maximized (see [this lecture note](http://www.eg.bucknell.edu/~phys310/skills/data_analysis/mle_intro.pdf) on taking derivative to find maximum) when the model parameter $$\theta$$ is the weighted average

$$
    \theta = \bar{I} = \arg\max p_\theta(I) 
            = \frac{\sum_i I_i/\sigma^2_{I_i}}{\sum_i 1/\sigma^2_{I_i}}.
$$

Variational inference (VI), in contrast, jointly performs scaling and merging by considering $$F$$ as a latent variable that generates the observed $$I$$. For simplicity, let's ignore the scaling aspect of the problem for now. With the introduction of latent variables, the likelihood term generally becomes intractable (For a tractable case, see [this post](https://towardsdatascience.com/mle-map-and-bayesian-inference-3407b2d6d4d9)):

$$
    p_\theta(I) = \int p_\theta(I,F)dF 
    = \int p_\theta(I|F)p(F)dF = \mathop{\mathbb{E}}_{p(F)}[p_\theta(I|F)].
$$
(Note that now $$\theta$$ refers to parameters in general rather than just the underlying structure factors. You can also rewrite the MLE equation as maximizing $$p_\theta(I|\theta)$$, which is the likelihood. In **m**aximum **a** **p**osteriori (MAP), the goal is to maximize the posterior, which is $$p_\theta(\theta|I) \propto p_\theta(I|\theta)p(\theta)$$. When you don't care about the prior $$p(\theta)$$, MAP becomes MLE.)

A naive estimation by Monte Carlo sampling from the prior distribution $$p(F)$$ converges slowly. A statistical trick that can be useful here is importance sampling, which samples from a surrogate distribution $$q(F)$$ and corrects the sampling bias:

$$
\begin{aligned}
    \int p_\theta(I|F)p(F)dF 
    &= 
    \int p_\theta(I|F)\frac{p(F)}{q(F)}q(F)dF \\
    &= \mathop{\mathbb{E}}_{q(F)}\left[p_\theta(I|F)\frac{p(F)}{q(F)}\right].
\end{aligned}
$$

Using Jensen's inequality, 

$$
\begin{aligned}
    \log p_\theta(I) &= \log \mathop{\mathbb{E}}_{q(F)}\left[p_\theta(I|F)\frac{p(F)}{q(F)}\right] \\
    &\geq \mathop{\mathbb{E}}_{q(F)}\left[\log\left[p_\theta(I|F)\frac{p(F)}{q(F)}\right]\right] \\
    &= \mathop{\mathbb{E}}_{q(F)}\left[\log p_\theta(I|F)\right] -\mathop{\mathbb{E}}_{q(F)}\left[\log \frac{q(F)}{p(F)}\right] \\
    &= \mathop{\mathbb{E}}_{q(F)}\left[\log p_\theta(I|F)\right] - D_\mathrm{KL}\left[q(F)\|p(F)\right]
\end{aligned}
$$

where the last line is also known as the **e**vidence-**l**ower **bo**und (ELBO). 
In other words, in VI, maximizing ELBO is guaranteed to maximize the likelihood. When $$q(F)=p(F|I)$$, $$p_\theta(I)$$ is exactly recovered using Bayes' theorem. In fact, ELBO can be alternatively derived from minimizing $$D_\mathrm{KL}[q(F)\|p(F|I)]$$. This motivates the interpretation of $$q(F)$$ as a good approximation of the intractable true posterior $$p_\theta(F|I)$$. Thus, unlike MLE or maximum a posteriori methods that yield a point estimation of $$F$$, VI enables characterizing the distribution of $$F$$ with uncertainty information. 

Another intuitive interpretation of maximizing ELBO as the optimization objective of VI is that the former term encourages fitting variable $$F$$ to observed data $$I$$ by maximizing the log-likelihood, whereas the latter Kullback–Leibler (KL) term penalizes overfitting by constraining model $$q(F)$$ to not deviate too far from the prior distribution $$p(F)$$.