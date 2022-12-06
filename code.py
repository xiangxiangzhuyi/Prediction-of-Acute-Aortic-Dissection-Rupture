import pandas as pd
import numpy as np
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def mul_classifier(tr_fl, te_fl, cal_num):
    classifiers = [
        SVC(kernel="linear", C=0.025, probability=True),
        LogisticRegression(),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]

    clf = classifiers[cal_num]
    clf.fit(tr_fl[:, 0:-1], tr_fl[:, -1])
    te_l = clf.predict(te_fl[:, 0:-1])

    te_acc = accuracy_score(te_l, te_fl[:, -1])
    te_pred = clf.predict_proba(te_fl[:, 0:-1])

    # spec and sen
    tn, fp, fn, tp = confusion_matrix(te_fl[:, -1], np.argmax(te_pred, -1)).ravel()
    sen = tp/(tp + fn)
    spe = tn/(tn + fp)

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(te_fl[:, -1], te_pred[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)

    precision = tp/(tp+fp)
    f1c = 2*tp/(2*tp + fp + fn)
    recall = sen

    return [auc, sen, te_acc, precision, f1c, spe, recall]


# read data
fi_na = 'E:/Project99_others/data/jiaceng_data/data.xlsx'
ex_da = pd.read_excel(fi_na, sheet_name=1, header=0)
np_da = np.array(ex_da)
fea, lab = np_da[:, 2:], np_da[:, 1]

# disorder the data
index = np.arange(len(lab))
random.shuffle(index)
fea, lab = fea[index], lab[index]

# divide into training and test
sam_num = 0.5*len(lab)
tr_fea, tr_lab = fea[ :sam_num], lab[ :sam_num]
te_fea, te_lab = fea[sam_num: ], lab[sam_num: ]
tr_fl = np.concatenate([tr_fea, tr_lab[:, np.newaxis]])
te_fl = np.concatenate([te_fea, te_lab[:, np.newaxis]])

# classification
all_met = []
for i in range(3):
    met = mul_classifier(tr_fl, te_fl, i)

all_met = np.array(all_met)


