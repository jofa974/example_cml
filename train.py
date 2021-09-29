import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, plot_confusion_matrix, roc_curve

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# Fit a model
depth = 5
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train,y_train)

fpr, tpr, _ = roc_curve(y_test, clf.predict(X_test))

metrics = {"accuracy" : clf.score(X_test, y_test),
           "AUC": auc(fpr, tpr)}
with open("metrics.json", 'w') as outfile:
        json.dump(metrics, outfile)


# Plot it
disp = plot_confusion_matrix(clf, X_test, y_test, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')



