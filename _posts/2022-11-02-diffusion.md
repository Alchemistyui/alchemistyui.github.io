---
title: Diffusion Models
layout: post
post-image: /assets/images/sites/sea2.jpg
description: A summary of diffusion models and some related resources.
tags:
- diffusion
- summary
---

# Background - Generative Models v.s. Discriminative Models

In machine learning, supervised learning can be divided into two kinds: Generative and discriminative. Generative models can generate new data instances by capturing the joint probability $p(X, Y)$, while discriminative models discriminate between different kinds of data instances by capturing the conditional probability $p(Y\| X)$.

Say that we want to solve a classification problem ($C_1, C_2, \dots, C_I$), for a new sample $x$, we want to predict its category by calculating the maximum conditional probability $p(y \| x)$.


### Generative Models

Generative Models have to model the distribution throughout the data space. The key point is to first assume the probability distribution of the data; then obtain statistic value for this distribution; and finally calculate the probability of a certain sample for different classes and get the prediction by the Bayesian formula.

Usually we suppose the data distribution is Gaussian distribution, because it's easy to calculate and is abundant in nature world. For $C_i$, the mean $\mu_i$ and variance $\Sigma_i$ of its Gaussian distributions is computed by some certain methods (like Maximum Likelihood Estimate). After having class data distributions, the probability that a sample $x$ belongs to class $C_i$ is: 

$$
P\left(x | C_i\right) = N(\mu_i, \Sigma_i) =\frac{1}{(2 \pi)^{D / 2}} \frac{1}{\left|\Sigma_i\right|^{1 / 2}} \exp \left\{-\frac{1}{2}\left(x-\mu_i\right)^T \Sigma_i^{-1}\left(x-\mu_i\right)\right\}
$$

where $D$ is the dimension of $x$. Now we have $P(C_i)$ and $P(x\|C_i)$, according to the Bayesian formula, the probability that $x$ belongs to class $C_i$ can be computed as:

$$
P(C_i|x) = \frac{P(x, C_i)}{P(x)} = \frac{P(x|C_i)P(C_i)}{\Sigma_{j=1}^{I} P(x|C_j)P(C_j)}
$$

Finally choose the one with maximum probability as the predicted class for the given sample $x$.

### Discriminative Models

Discriminative models directly solve the conditional probability $P(C_i\|x)$ by building a network. Then train this network to get the proper value for model weight. They cannot reflect the characteristics of the training data themselves, and only tell us the classification information, so they are more limited.


# What are diffusion models?

# Highlighted Works

## DDPM

## Improved DDPM

## Stable Diffusion







#### References:

[1] [DDPM: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

[2] [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process)

[3] [Explaination for Improved DDPM](https://www.youtube.com/watch?v=gwI6g1pBD84&list=PL1v8zpldgH3pXjOUhfPVH3EhW4WMHVYPh)

[4] [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/abs/2209.00796)