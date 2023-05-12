import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import metrics
from numpy import mean
x_train=np.loadtxt('x_train.txt')
y_train=np.loadtxt('y_train.txt')
x_test=np.loadtxt('x_test.txt')
y_test=np.loadtxt('y_test.txt')
x_val=np.loadtxt('x_val.txt')
y_val=np.loadtxt('y_val.txt')
def constructSVM():
    from sklearn import svm
    from sklearn.svm import SVC
    svc = svm.SVC(kernel ='rbf') # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    return svc
se_list = []
sp_list = []
precise_list = []
acc_list = []
f1_list = []
mcc_list = []
auc_list = []
ap_list = []
cv = StratifiedKFold(n_splits=5)
for fold, (train_index, test_index) in enumerate(KFold(n_splits=5, shuffle=True).split(x_train)):
    print('fold%r' % fold)
    new_x_train, new_y_train = x_train[train_index], y_train[train_index]
    new_x_test, new_y_test = x_train[test_index], y_train[test_index]
    test_label = y_train[test_index]
    model = constructSVM()
    model.fit(new_x_train, new_y_train)
    pred = model.predict(new_x_test)
    pred_res = model.predict_proba(new_x_test)[:,1]
    pred_label = [0 if x < 0.5 else 1 for x in pred_res]
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=test_label, y_pred=pred_label).ravel()
    recall = metrics.recall_score(y_pred=pred_label, y_true=test_label)
    precise = metrics.precision_score(y_pred=pred_label, y_true=test_label)
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = metrics.accuracy_score(y_pred=pred_label, y_true=test_label)
    f1 = metrics.f1_score(y_pred=pred_label, y_true=test_label)
    mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=test_label)
    auc = metrics.roc_auc_score(y_true=test_label, y_score=pred_res)
    ap = metrics.average_precision_score(y_score=pred_res, y_true=test_label)
    se_list.append(se)
    sp_list.append(sp)
    precise_list.append(precise)
    acc_list.append(acc)
    f1_list.append(f1)
    mcc_list.append(mcc)
    auc_list.append(auc)
    ap_list.append(ap)
    print("se：", se, "sp：", sp, "precise：", precise, "acc：", acc, "f1：", f1, "mcc：", mcc, "auc：", auc, "ap：", ap)
print('交叉验证的平均指标如下：')
print("se：", mean(se_list))
print("sp：", mean(sp_list))
print("acc：", mean(acc_list))
print("precise：", mean(precise_list))
print("F1：", mean(f1_list))
print("mcc：", mean(mcc_list))
print("auc：", mean(auc_list))
print("ap：", mean(ap_list))
