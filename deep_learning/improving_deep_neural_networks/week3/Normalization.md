---
title: "Normalizing activations in a network"
author: Chinmay Hundekari
date: May 07, 2020
---

# To execute:
pandoc Normalization.md -f markdown+tex_math_dollars -s -o Normalization.html --mathjax --toc -V toc-title:"Table of Contents"

# Normalizing activations in a network
## Batch Normalization
* Easily train deep networks.
Normalizing inputs helps to speed up learning. So, can we normalize z() or a() to speed up that layer.

* Normalize each layer of z separately
$$ \mu = \frac{1}{m} \sum\limits_{i} z^i  $$
$$ \sigma^{2} = \frac{1}{m} \sum\limits_{i} (z^i - \mu)^{2} $$
$$ z^{i}_{norm} = \frac{z^{i} - \mu}{\sqrt{\sigma + \epsilon}}  $$
$$ \widetilde{z}^{i} = \gamma z^{i}_{norm} + \beta  $$, where $\gamma$ and $\beta$ are learnable parameters.

* This would help in taking advantage of the non-linear region of sigmoid function, or any other activation function.

## Fitting Batch Norm into a neural network
### Feedforward Propagation
1. Calculate $z^{[l]}$ for a layer. 
2. Calculate batch norm $\widetilde{z}^{[l]}$ for $z^{[l]}$, with $\beta^{[l]}$ and $\gamma^{[l]}$.
3. Use $\widetilde{z}^{[l]}$ to calclulate $a^{[l]}$.
4. Repeat for all layers.

### BackPropagation
1. Back propagate $\w^{[l]}$ changes with equation $\w^{[l]} = \w^{[l]} - \alpha d\w^{[l]}$.
2. Back propagate $\beta^{[l]}$ changes with equation $\beta^{[l]} = \beta^{[l]} - \alpha d\beta^{[l]}$.
3. Back propagate $\gamma^{[l]}$ changes with equation $\gamma^{[l]} = \gamma^{[l]} - \alpha d\gamma^{[l]}$.

* Equation $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$ changes to $z^{[l]} = W^{[l]}a^{[l-1]}$. This is because $z$ is normalized, and the averaging leads to loss of $b$. This role is now taken by $\beta$.
