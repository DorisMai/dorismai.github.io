---
title: 'Inference time guidance for generative models'
date: 2026-01-11
modified: 2026-01-11
permalink: /posts/inference-time-guidance/
color_tag: "study-notes"
tags:
  - ML
---

## Background
At inference time of a diffusion model, the plain algorithm (assuming model trained by [score-matching](https://arxiv.org/pdf/1907.05600)) from the perspective of reconstruction is:
 
> **Input:** Diffusion noise schedule $$\sigma(t)$$, Langevin noise schedule $$\eta(t)$$, Trained score model $$s_\theta(x_t, t)$$
>
> **Output:** Sample $$x_0$$
>
> 1.  Sample $$x_T \sim \mathcal{N}(0, I)$$  
> 2.  **for** $$t = T, T-1, \dots, 1$$ **do**
> 3.    $$\hspace{1em}$$ $$z \sim \mathcal{N}(0, I)$$
> 4.    $$\hspace{1em}$$ $$\hat{s} \leftarrow s_\theta(x_t, t)$$
> 5.    $$\hspace{1em}$$ $$\hat{x}_0 = \mathrm{GetPredictedCleanSample}(x_t, \hat{s}, \sigma(t))$$
> 6.    $$\hspace{1em}$$ $$x_{t-1} = \mathrm{WeightedAverage}(\hat{x}_0, x_t, \sigma(t)) + \eta(t) z$$  $$\hspace{1em}$$
> 7.  **end for**
> 8. **return** $$x_0$$
{: .pseudocode }

Note that training on predicting the added noise ($$\hat{\epsilon}_\theta$$) or the clean sample ($$\hat{x}_0$$) is equivalent to (i.e. mathematically interconvertible) predicting the conditional score $$\nabla_{x_t} \log p(x_t \mid x_0)$$. 

## Guidance
Given an observable $$y$$, a forward model $$y = A(x) + \epsilon$$, guide the inference process to solve the inverse problem that recovers $$x$$ from $$y$$ using the trained model as a prior. $$\epsilon$$ practically represents an error model for the observable, such as normal distribution, student-t distribution, etc. 

The basic idea of guidance is following Bayes rule:

$$\nabla_{x_t} \log p_t(x_t\mid y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p(y\mid x_t)$$

### DPS
In [DPS (Denoising Posterior Sampling)](https://arxiv.org/abs/2209.14687), the algorithm is:
> **Input:** Guidance schedule $$\zeta(t)$$, Observable $$y$$, Forward model $$A$$ and error model $$\epsilon(t)$$, Diffusion noise schedule $$\sigma(t)$$, Langevin noise schedule $$\eta(t)$$, Trained score model $$s_\theta(x_t, t)$$
>
> **Output:** Sample $$x_0$$
>
> 1.  Sample $$x_T \sim \mathcal{N}(0, I)$$  
> 2.  **for** $$t = T, T-1, \dots, 1$$ **do**
> 3.    $$\hspace{1em}$$ $$z \sim \mathcal{N}(0, I)$$
> 4.    $$\hspace{1em}$$ $$\hat{s} \leftarrow s_\theta(x_t, t)$$
> 5.    $$\hspace{1em}$$ $$\hat{x}_0 = \mathrm{GetPredictedCleanSample}(x_t, \hat{s}, \sigma(t))$$
> 6.    $$\hspace{1em}$$ $$\hat{p}(y\mid x_t) \leftarrow \mathrm{ComputeLikelihood}(y, A, \hat{x}_0, \epsilon(t))$$
> 7.    $$\hspace{1em}$$ $$x_{t-1} = \mathrm{WeightedAverage}(\hat{x}_0, x_t, \sigma(t)) + \eta(t) z - \zeta(t) \nabla_{x_t} \log \hat{p}(y\mid x_t)$$  
> 8.  **end for**
> 9. **return** $$x_0$$
{: .pseudocode }

The error model is time-step dependent mostly for engineering purposes, as the effect of introducing the likelihood term is difficult to control. This is because the forward model is usually defined only for the clean sample, so we know $$p(y\mid x_0)$$ but not $$p(y\mid x_t)$$. DPS and many other paper addresses this issue by:

$$
\begin{aligned}
    p(y\mid x_t) &= \int p(y\mid x_0) p(x_0\mid x_t) dx_0 \\
    &= \mathrm{E}_{x_0 \sim p(x_0\mid x_t)}[p(y\mid x_0)] \\
    &\approx p(y\mid x_0) \hspace{1em} \text{point estimation}
\end{aligned}
$$
The likelihood approximation works using Tweedie's formula which shows that $$\hat{x}_0 = \mathrm{E}[x_0\mid x_t]$$.

There are some major risks associated with the this type of guidance approach:
- The point estimation of the likelihood can have huge variance and affects the guidance/denoising process.
- The introduced likelihood term might push the sample away from the true data manifold.


### PnP
A harder way to enforce consistency with obersable is using [plug-and-play (PnP) guidance](https://arxiv.org/pdf/2305.08995). The main difference is PnP directly minimizes $$\hat{x}_0$$ with respect to likelihood loss rather than taking a small gradient step like in DPS.

> **Input:** Noise balance schedule $$\zeta(t)$$, Observable $$y$$, Forward model $$A$$ and error model $$\epsilon(t)$$, Diffusion noise schedule $$\sigma(t)$$, Langevin noise schedule $$\eta(t)$$, Trained score model $$s_\theta(x_t, t)$$
>
> **Output:** Sample $$x_0$$
>
> 1.  Sample $$x_T \sim \mathcal{N}(0, I)$$  
> 2.  **for** $$t = T, T-1, \dots, 1$$ **do**
> 3.    $$\hspace{1em}$$ $$z \sim \mathcal{N}(0, I)$$
> 4.    $$\hspace{1em}$$ $$\hat{s} \leftarrow s_\theta(x_t, t)$$
> 5.    $$\hspace{1em}$$ $$\hat{x}_0 = \mathrm{GetPredictedCleanSample}(x_t, \hat{s}, \sigma(t))$$
> 6.    $$\hspace{1em}$$ $$\hat{x}_0^* = \operatorname{argmin}_x \mathrm{LikelihoodLoss}(y, A, x, \epsilon(t)) + \mathrm{L2Constraint}(x, \hat{x}_0)$$
> 7.    $$\hspace{1em}$$ $$x_{t-1} = \mathrm{WeightedAverage}(\hat{x}_0^*, x_t, \sigma(t), \zeta(t)) + \eta(t) z$$  
> 8.  **end for**
> 9. **return** $$x_0$$
{: .pseudocode }

A latent space variant of PnP is [here](https://openreview.net/pdf?id=j8hdRqOUhN).
A specific way of minimizing the likelihood loss especially for non-linear or non-differentable problem is using [pseudo-inverse guidance](https://openreview.net/pdf?id=9_gsMA8MRKQ).

## SMC
A popular approach in guidance is to use sequential Monte Carlo (SMC) for particle-filtering rather than denoising a single sample. SMC is closely related to Annealed Importance Sampling (AIS). This might be a way to mitigate the issue of the high variance associated with the point estimation of the likelihood.

### TDS
[Twisted diffusion sampling (TDS)](https://arxiv.org/pdf/2306.17775) algorithm is as follows:

> **Input:** Number of particles $$K$$, Observable $$y$$, likelihood $$p(y\mid x)$$, Diffusion noise schedule $$\sigma(t)$$, Langevin noise schedule $$\eta(t)$$, Trained score model $$s_\theta(x_t, t)$$
>
> **Output:** Sample $$\{x_0^k\}_{k=1}^K$$
>
> 1. **for** $$k = 1, 2, \dots, K$$ **do** $$\hspace{1em}$$ # Initialize particles
> 2.    $$\hspace{1em}$$ Sample $$x_T^{(k)} \sim \mathcal{N}(0, I)$$  
> 3.    $$\hspace{1em}$$ $$w_k^T \leftarrow p(y\mid x_T^{(k)} \approx \hat{x}_0^k)$$
> 4. **for** $$t = T-1, \dots, 1$$ **do** $$\hspace{1em}$$ # Denoising steps
> 5.    $$\hspace{1em}$$ $$\{x_k^{t+1}\}_{k=1}^K \sim \mathrm{Multinomial}(\{w_k^{t+1}\}_{k=1}^K)$$ $$\hspace{1em}$$ # resample
> 6.    $$\hspace{1em}$$ **for** $$k = 1, 2, \dots, K$$ **do**
> 7.    $$\hspace{2em}$$ $$\hat{s}^k \leftarrow s_\theta(x_k^{t+1}, t)$$
> 8.    $$\hspace{2em}$$ $$\hat{x}_0^k = \mathrm{GetPredictedCleanSample}(x_k^{t+1}, \hat{s}, \sigma(t))$$
> 9.    $$\hspace{2em}$$ $$\hat{s}_y^k \leftarrow \hat{s}^k + \nabla_{x_t} \log p(y\mid x_{t+1}^k \approx \hat{x}_0^k)$$
> 10.    $$\hspace{2em}$$ $$x_{k}^{t} = \mathrm{GaussianTransition}(\hat{s}_y^k, x_k^{t+1}, \sigma(t))$$ $$\hspace{1em}$$ # propose
> 11.    $$\hspace{2em}$$ $$w_k^t \leftarrow \frac{\mathrm{GaussianTransition}(\hat{s}^k, x_k^{t+1}, \sigma(t))}{\mathrm{GaussianTransition}(\hat{s}^k_y, x_k^{t+1}, \sigma(t))} \frac{p(y\mid x_t^k \approx \hat{x}_0^k)}{p(y\mid x_{t+1}^k \approx \hat{x}_0^k)}$$ $$\hspace{1em}$$ # reweight
> 12.    $$\hspace{1em}$$ **end for**
> 13. **end for**
> 14. **return** $$\{x_0^k\}_{k=1}^K$$
{: .pseudocode }

The Gaussian transition kernel is essentially the reverse diffusion step that depends on the predicted score and the noise schedule. The guided transition simply modifies the predicted score according the Bayes' rule like in DPS.

The reweight step is same thing in importance sampling, where the denominator is the distribution sampled from (using the guided gaussian transition + initial sampling of the particles), and the nominator is the distribution we want to sample from (using the original denoising step + the new likelihood).

### FK
[Feyman-Kac (FK) steering](https://arxiv.org/pdf/2501.06848) is a more general framework that can accomodate other methods like TDS.

The algorithm is very similar:
> **Input:** Number of particles $$K$$, Observable $$y$$, Potentials $$G_t$$, proposal kernel $$\tau(x_t\mid x_{t+1}, y)$$, Diffusion noise schedule $$\sigma(t)$$, Langevin noise schedule $$\eta(t)$$, Trained score model $$s_\theta(x_t, t)$$
>
> **Output:** Sample $$\{x_0^k\}_{k=1}^K$$
>
> 1. **for** $$k = 1, 2, \dots, K$$ **do** $$\hspace{1em}$$ # Initialize particles
> 2.    $$\hspace{1em}$$ Sample $$x_T^{(k)} \sim \mathcal{N}(0, I)$$  
> 3.    $$\hspace{1em}$$ $$G_T^{(k)} \leftarrow G_T(x_T^{(k)})$$
> 4. **for** $$t = T-1, \dots, 1$$ **do** $$\hspace{1em}$$ # Denoising steps
> 5.    $$\hspace{1em}$$ $$\{x_k^{t+1}\}_{k=1}^K \sim \mathrm{Multinomial}(\{w_k^{t+1}\}_{k=1}^K)$$ $$\hspace{1em}$$ # resample
> 6.    $$\hspace{1em}$$ **for** $$k = 1, 2, \dots, K$$ **do**
> 7.    $$\hspace{2em}$$ $$x_{k}^{t} = \tau(x_k^t\mid x_{k}^{t+1}, y)$$ $$\hspace{1em}$$ # propose
> 9.    $$\hspace{2em}$$ $$G_t^k \leftarrow \frac{\mathrm{GaussianTransition}(\hat{s}^k, x_k^{t+1}, \sigma(t))}{\tau(x_k^t\mid x_{k}^{t+1}, y)} G_t^k$$ $$\hspace{1em}$$ # reweight
> 10.    $$\hspace{1em}$$ **end for**
> 11. **end for**
> 12. **return** $$\{x_0^k\}_{k=1}^K$$
{: .pseudocode }

This framework recovers TDS when (a) the proposal kernel uses the guided gaussian transition, and (b) the potential $$G_t := \exp(r(x_t) - r(x_{t+1}))$$ is used, where the reward function $$r(x_t) = \log p(y\mid x_t \approx \hat{x}_0)$$ is used.

The paper discusses other choices of implementation details, such as choices of the proposal kernel, the potential function, and the reward function. Nothing drastically deviating from the paradigm discussed above.

