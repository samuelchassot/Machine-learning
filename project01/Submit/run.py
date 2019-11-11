# -*- coding: utf-8 -*-

from implementations import *
from proj1_helpers import *

print("import train.csv")
DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

print("import test.csv")
DATA_TEST_PATH = 'data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print("clean data")
tX = remove_wrong_columns(tX)
tX_test = remove_wrong_columns(tX_test)

print("standardize tX and tX_test")
tX_stdzed, tX_test_stdzed = standardize_train_and_test(tX, tX_test)

# this implementation of ridge regression provided us the
# following result: 0.765 0.627
lambda_ = 0.00035564803062231287
degree  = 5

print("expand features tX and tX_test")
poly_tX_stdzed = expand_features_polynomial(tX_stdzed, degree)
poly_tX_test_stdzed = expand_features_polynomial(tX_test_stdzed, degree)

print("compute weights with ridge ridge_regression")
w, _ = ridge_regression(y, poly_tX_stdzed, lambda_)

print("create submission")
OUTPUT_PATH = 'data/J_D_S_best_submission.csv'
y_pred = predict_labels(w, poly_tX_test_stdzed)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
