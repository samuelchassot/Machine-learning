# README

The parameters degree and lambda\_ used for ridge regression were derived from multiple cases analysis. The code can be found as a jupyter notebook _ridge.ipynb_.

## Data preparation
First, we standardize the data, then, we apply basic data cleaning by removing feature where thera are -999 values.

## Feature generation
We compared multiple degrees of polynomial feature expansion starting from 1 up to 20 with lambda\_s on a 50 value logscale going from 10^-30 to 10^0. We balanced underfitting and overfitting by taking care of maintaining the value of lambda\_ _not to small and not to big_ and looking at the sweet spot minimizing the mean squared error. Then, we fine-grained the values around the sweet spot to find the most accurate minimum.

## cross-validation steps
We splitted the train set into five distinct subsets to perform cross-validation.
