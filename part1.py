import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import statistics

datasets = ["tictac_final.txt", "tictac_single.txt"]  # , "tictac_multi.txt"]
np.set_printoptions(precision=4)


def score_cm(clf, X, y):
    # using 10-Fold cross validation
    scores = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        # clf.decision_function([[1]])
        y_pred = clf.predict(X_test)
        y_pred = y_pred > 0.5
        scores.append(accuracy_score(y_test, y_pred))
        # score = 0

    # mean and std
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (statistics.mean(scores), statistics.stdev(scores)))
    return clf


def score_cm2(clf, X, y):
    # using 10-Fold cross validation
    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)
    # mean and std
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    # confusion matrices
    y_pred = cross_val_predict(clf, X, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)
    print("confusion matrix is:\n", conf_mat)
    sum_of_rows = conf_mat.sum(axis=1)
    norm_cm = conf_mat / sum_of_rows[:, np.newaxis]
    print("After normalizing each row to 1:\n", norm_cm)


## classifier
for dt in datasets:
    print("### dataset ###", dt)
    A = np.loadtxt(dt)
    X = A[:, :9]
    y = np.squeeze(A[:, 9:])
    print(X.shape)
    print(y.shape)
    # shuffle data
    random_state = np.random.RandomState(0)
    X, y = shuffle(X, y, random_state=random_state)

    # SVM
    print("## Linear SVC ##")
    clf_svm = LinearSVC(random_state=random_state, max_iter=10000)
    # clf_svm.decision_function([[1]])
    score_cm2(clf_svm, X, y)

    # KNN
    print("## KNN ##")
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    score_cm2(clf_knn, X, y)

    # MLP
    print("## MLP ##")
    clf_mlp = MLPClassifier(random_state=random_state, max_iter=10000)
    score_cm2(clf_mlp, X, y)

## regressor
# ## multi-label dataset
dt = "tictac_multi.txt"
print("### dataset ###", dt)
A = np.loadtxt(dt)
X = A[:, :9]
y = np.squeeze(A[:, 9:])
print(X.shape)
print(y.shape)
# shuffle data
random_state = np.random.RandomState(0)
X, y = shuffle(X, y, random_state=random_state)

# KNN
print("## KNN ##")
clf_knn = KNeighborsRegressor(n_neighbors=5)
# # clf_knn = MultiOutputClassifier(KNeighborsClassifier())
# clf_knn = OneVsRestClassifier(KNeighborsClassifier())
score_cm(clf_knn, X, y)

# Linear Regressor
# train model for each output label
from numpy.linalg import inv

# using 10-Fold cross validation
print("## Linear Regressor ##")
scores = []
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # solving linear least squares
    b = inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    # print(b)
    # predict using coefficients
    y_pred = X_test.dot(b)
    y_pred = y_pred > 0.2
    scores.append(accuracy_score(y_test, y_pred))

# mean and std
print("Accuracy: %0.2f " % (sum(scores) / len(scores)))

# MLP
print("## MLP ##")
clf_mlp = MLPRegressor(hidden_layer_sizes=[100, 500, 100], random_state=random_state, max_iter=10000)
clf_save = score_cm(clf_mlp, X, y)

# save model
import pickle

filename = "model.pkl"
pickle.dump(clf_save, open(filename, "wb"))

# # split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# clf_svm.fit(X_train,y_train)
# # y_score = clf_svm.decision_function(X_test,y_test)
# # predict
# pred = clf_svm.predict(X_test)
# print(pred)
# score
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test,pred)
# print(acc)

# conf_matrix_list_of_arrays = []
# kf = cross_validation.KFold(len(y), n_folds=5)
# for train_index, test_index in kf:
#
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#
#    model.fit(X_train, y_train)
#    conf_matrix = confusion_matrix(y_test, model.predict(X_test))
#    conf_matrix_list_of_arrays.append(conf_matrix)
