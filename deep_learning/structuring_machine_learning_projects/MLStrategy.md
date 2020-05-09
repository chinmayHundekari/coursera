---
title: "Machine learning Strategy"
author: Chinmay Hundekari
date: May 07, 2020
---

# To execute:
~~~~
pandoc MLStrategy.md -f markdown+tex_math_dollars -s -o MLStrategy.html --mathjax --toc -V toc-title:"Table of Contents"
~~~~

# Orthogonalization
* 1 variable to be tuned with 1 purpose, simplifies tuning. 
* Orthogonal means at $90\deg$ to each other. Each parameter is independent of others.
* Chain of assumptions in ML
1. Fit training set well on cost function  --> Bigger network, ADAM, better optimizer
2. Fit dev set well on cost functions. --> Regularization, Bigger training set
3. Fit test set well on cost function. --> Bigger dev set
4. Performs well in the world. --> Change dev set or cost function.
* Early stopping is less orthogonal, as it affects multiple features, hence not preferred.

# Single number evaluation metric
* 
