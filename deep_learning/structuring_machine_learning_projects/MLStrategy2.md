---
title: "Machine learning Strategy 2"
author: Chinmay Hundekari
date: May 10, 2020
---

# To execute:
~~~~
pandoc MLStrategy2.md -f markdown+tex_math_dollars -s -o MLStrategy2.html --mathjax --toc -V toc-title:"Table of Contents"
~~~~

# Error analysis
* Estimate improvement that can be achieved before putting in effort to fix a classification error.
* Evaluating multiple ideas/improvements/errors in parallel, to reduce effort spent on training.
* Incorrectly labelled examples may not be a big issue, if the data set is large, or the effort to correct the labelling is costly. 
* Build quickly and then iterate
    1. Set up a dev/test set metric
    2. Build initial system quickly.
    3. Use Error analysis to figure out what preprocessing needs to be added.
    4. Use Bias/variance analysis to reduce error.

# Mismatched training and dev/test set error
* Larger amounts of data needed for deep learning encourages to get training data outside dev/test data distribution
* Adding addition data to training distribution is ok, but having test data equally to distributed between train, dev and test data is not ideal. Instead retain a higher concentration of original test data in test set.
* Split the training set into training and training-dev set. Only train on training set. 
    1. If training error >> Bayer error, then there is a bias problem. 
    2. If dev error ~= training-dev error >> training error, then there is a variance problem. 
    3. If dev error >> training-dev error ~= training error, then there is a data mismatch issue.
    4. If dev error < training-dev error ~= training error, then there is a data mismatch issue.
* Use data synthesis to fix data mismatch error.

