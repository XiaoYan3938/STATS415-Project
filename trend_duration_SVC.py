from sklearn import svm
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import savefig


def input_data(filename):
    X = []
    y = []
    with open(filename, "r") as f:
        trend_or_not = []
        t_period = []
        appeared_times = []
        reader = csv.DictReader(f)
        for row in reader:
            trend_or_not.append(row["will_end_trend_at_period_t+1_or_not"])
            t_period.append(float(row["t_period_log_views_per_day_percentage_change"]))
            appeared_times.append(float(row["appeared_times"]))
        X = np.hstack((np.array([t_period]).T, np.array([appeared_times]).T))
        for bool in trend_or_not:
            if bool == "False":
                y.append(0)
            else:
                y.append(1)
    return X, y


def train_SVC(X, y, C=1, probability=False):
    clf = svm.SVC()
    clf.C = C
    clf.kernel = "rbf"
    clf.gamma = "auto"
    clf.probability = probability
    clf.fit(X_train, y_train)
    return clf


def score_SVC(X, y, clf):
    pre_y =clf.predict(X)

    one_count = 0
    correct_count = 0
    correct_one_count = 0
    for y_i, pre_y_i in zip(y, pre_y):
        if y_i == 1:
            one_count += 1
        if  y_i == pre_y_i:
            correct_count += 1
            if y_i == 1:
                correct_one_count += 1
    return correct_one_count/one_count, correct_count/len(y)


X_train, y_train = input_data("US_videos_SVC.csv")
X_test, y_test = input_data("GB_videos_SVC.csv")

# clf = train_SVC(X_train, y_train, 5, True)
# fix_lvpc = [[-1, 1], [-1, 3], [-1, 5], [-1, 7], [-1, 9]]
# fix_times = [[-1, 5], [-3, 5], [-5, 5], [-7, 5], [-9, 5]]
# print(clf.predict_proba(fix_lvpc))
# print(clf.predict_proba(fix_times))
# # [[0.83372619 0.16627381]
# #  [0.83375936 0.16624064]
# #  [0.83351767 0.16648233]
# #  [0.76287779 0.23712221]
# #  [0.83367442 0.16632558]]
# # [[0.83351767 0.16648233]
# #  [0.83367269 0.16632731]
# #  [0.83370457 0.16629543]
# #  [0.8052437  0.1947563 ]
# #  [0.89933604 0.10066396]]

# clf = train_SVC(X_train, y_train, 0.5)
# print(score_SVC(X_test, y_test, clf))
# # (0.018997197134848955, 0.904281098546042)
# clf = train_SVC(X_train, y_train, 1)
# print(score_SVC(X_test, y_test, clf))
# # (0.03581438804110869, 0.8976575121163166)
# clf = train_SVC(X_train, y_train, 5)
# print(score_SVC(X_test, y_test, clf))
# # (0.04640298972282778, 0.8929186860527732)
# clf = train_SVC(X_train, y_train, 10)
# print(score_SVC(X_test, y_test, clf))
# # (0.05076300218000623, 0.891787829833064)
# clf = train_SVC(X_train, y_train, 20)
# print(score_SVC(X_test, y_test, clf))
# # (0.054500155714730616, 0.8913031771674744)
