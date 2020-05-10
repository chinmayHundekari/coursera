---
title: "Machine learning Strategy 1"
author: Chinmay Hundekari
date: May 07, 2020
---

# To execute:
~~~~
pandoc MLStrategy1.md -f markdown+tex_math_dollars -s -o MLStrategy1.html --mathjax --toc -V toc-title:"Table of Contents"
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
## Metrics
* Positive Samples - Part of the classification. 
* Negative Samples - Not part of the classification.
* True Positives - Correctly Identified Positive Samples
* False Positives - Incorrectly Identified Positive Samples
* True Negatives - Correctly Identified Negative Samples
* False Negatives - Incorrectly Identified Negative Samples
* Accuracy - 
$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$
* Accuracy may not accurately represent a class imbalanced data set, where TP are few.
* Precision - What percentage of recognized class were correctly recognized? or True Positives.
$$ Precision = \frac{TP}{TP + FP} $$
* Recall - What percentage of all positive samples were recognized? or True Positives/(All Positives)
$$ Recall = \frac{TP}{TP + FN} $
* Receiver Operating Characteristic Curve - Graph of TP Rate vs FP Rate
$$ True Postive Rate(TPR) = \frac{TP}{TP + FN}
$$ False Postive Rate(FPR) = \frac{FP}{FP + TN}
* F1 Score - Harmonic mean of P and R 
$$ F1 = \frac{2}{\frac{1}{P} + \frac{1}{R}} $$

* Satisficing metric - certain conditions should be satisfied. - e.g. running time should be less than 100ms.
* Optimizing metric - Certain conditions need to be as good aspossible - e.g. Accuracy.

# Training/dev/test distributions
* Dev and test sets should be from same distribution. Train set may not be same distribution, but needs to be similar.
* Purpose of test set is only to provide a high confidence.
* Deep learning algorithms require a lot of data, pushing training data to be maximum of all samples.
* Evaluation metric should not confuse Positives and Negatives. e.g. Scenario - Spam should not be allowed - Positives are spam
* Evaluation metric should not ignore Negatives which may have a bigger cost. e.g. Cat classifier for kids, may show FP on adult images. The metric should have a higher cost on certain FP.

# Comparing human level performance
* Bayes Optimal error - Lowest possible error rate for classification. Human-level error can be assumed to be Bayes error.
* Understanding Human level performance, hints us to know how much Bias-Variance tradeoff should be.
* Avoidable bias - Bayes error - Training error
* If Variance (Dev set error - Training error) is greater than Avoidable bias, then we should focus on reducing variance rather than bias.
*  Problems which are not natural perception tasks, tend to have much lesser Human-level performance than Bayes optimal error.

# Improving your model performance.
1. You can reduce training set error - Reduce avoidable bias.
    1. Train bigger model
    2. Training longer/better optimization algorithms.
    3. Different neural network architecture or better hyperparameter search
    4. Reduce regularization, if variance is low, but test set performance is bad.
2. You can reduce dev set error - Reduce variance.
    1. More data
    2. Regularization, L2, dropout, Data augumentation
    3. Different neural network architecture or better hyperparameter search
