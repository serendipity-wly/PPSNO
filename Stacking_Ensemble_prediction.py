
import numpy as np
from keras.layers import Dropout, MaxPooling1D, Flatten, Embedding, Dense, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D,Activation
from numpy import mean
from keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.models import Sequential
from tensorflow import keras
early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None
    )


callbacks = [early_stopping]

def trainensemble(x_train,y_train,x_val, y_val,x_test,y_test):
    from tensorflow import keras
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None
    )

    callbacks = [early_stopping]

    model2 = Sequential()
    model2.add(Embedding(input_dim=300, output_dim=128, input_length=7460, mask_zero=False))
    model2.add(Dropout(0.2))
    model2.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu'))
    model2.add(Activation(activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
    model2.add(Activation(activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(GlobalMaxPooling1D())
    model2.add(Dense(100))
    model2.add(Dropout(0.2))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=128, callbacks=callbacks,verbose=1)
    pred_res = model2.predict(x_train)
    pred_res2 = pred_res.flatten()
    pred_res_cnn = np.array(pred_res2).reshape(len(pred_res2), 1)
    a1 = np.array(pred_res_cnn)

    model = Sequential()
    model.add(Embedding(input_dim=300, output_dim=128, input_length=7460))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, input_shape=(7460, 128)))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, input_shape=(7460, 64)))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=500, callbacks=callbacks,verbose=1)
    pred_res3 = model.predict(x_train)
    pred_res3 = pred_res3.flatten()
    pred_res_lstm = np.array(pred_res3).reshape(len(pred_res3), 1)
    a8 = np.array(pred_res_lstm)

    from catboost import CatBoostClassifier
    c = CatBoostClassifier(iterations=2983,
                           learning_rate=0.03823,
                           depth=8,
                           subsample=0.7847,
                           rsm=0.8104,
                           task_type='CPU')
    c.fit(x_train, y_train)
    pred_res1 = c.predict_proba(x_train)[:, 1]
    pred_res_cat = np.array(pred_res1).reshape(len(pred_res1), 1)
    a2 = np.array(pred_res_cat)

    from sklearn import svm
    from sklearn.svm import SVC
    svc = svm.SVC(kernel ='rbf')
    svc.fit(x_train, y_train)
    pred_res_svm = svc.predict_proba(x_train)[:, 1]
    pred_res_svm = np.array(pred_res_svm).reshape(len(pred_res_svm), 1)
    a3 = np.array(pred_res_svm)

    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=4906,
                        colsample_bytree=0.961263998009275,
                        learning_rate=0.004264475482427766,
                        max_depth=13,
                        min_child_weight=45.48826116566274,
                        subsample=0.8556594922541692
                        )
    xgb.fit(x_train, y_train)
    pred_res_xgb = xgb.predict_proba(x_train)[:, 1]
    pred_res_xgb=np.array(pred_res_xgb).reshape(len(pred_res_xgb), 1)
    a4=np.array(pred_res_xgb)

    from sklearn.ensemble import GradientBoostingClassifier
    gbdt = GradientBoostingClassifier(loss='deviance',
                                      n_estimators=2000,
                                      learning_rate=0.028673784262499755,
                                      max_depth=11,
                                      min_samples_leaf=59,
                                      min_samples_split=19,
                                      subsample= 0.6447933070462003,
                                      max_features=0.0001
                                      )
    gbdt.fit(x_train, y_train)
    pred_res_gbdt = gbdt.predict_proba(x_train)[:, 1]
    pred_res_gbdt = np.array(pred_res_gbdt).reshape(len(pred_res_gbdt), 1)
    a5=np.array(pred_res_gbdt)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=3789,
                                max_depth=19,
                                min_samples_leaf=82,
                                min_samples_split=4,
                                bootstrap=True,
                                max_features='auto',
                                n_jobs=1,
                                random_state=3,
                                verbose=100,
                                oob_score=True
                                )

    rf.fit(x_train, y_train)
    pred_res_rf = rf.predict_proba(x_train)[:, 1]
    pred_res_rf=np.array(pred_res_rf).reshape(len(pred_res_rf), 1)
    a6=np.array(pred_res_rf)

    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=3835,
                             learning_rate=0.0925472963334393,
                             max_depth=13,
                             cat_smooth=30.61541228449972,
                             bagging_fraction=0.6582040629836386,
                             bagging_freq=9,
                             feature_fraction=0.6887307635842295,
                             lambda_l1=0.9911958420007507,
                             lambda_l2 =34.290859455476834
                             )
    clf.fit(x_train, y_train,eval_set=(x_val, y_val), eval_metric="auc")
    pred_res_lgb = clf.predict_proba(x_train)[:, 1]
    pred_res_lgb = np.array(pred_res_lgb).reshape(len(pred_res_lgb), 1)
    a7 = np.array(pred_res_lgb)

    pred_res_sum = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8), axis=1)
    return np.array(pred_res_sum)

def testensemble(x_train,y_train,x_val, y_val,x_test,y_test):
    from keras.layers import Dropout, MaxPooling1D, Flatten, Embedding, Dense, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D,Activation
    from keras.models import Sequential
    from tensorflow import keras
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None
    )
    callbacks = [early_stopping]
    model=Sequential()
    model.add(Embedding(input_dim=300, output_dim=128, input_length=7460))
    model.add(Dropout(0.2))
    model.add(LSTM(128,return_sequences=True,input_shape=(7460,128)))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, input_shape=(7460, 64)))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=50, epochs=200, callbacks=callbacks,verbose=1)
    pred_res3 = model.predict(x_test)
    pred_res3 = pred_res3.flatten()
    pred_res_lstm = np.array(pred_res3).reshape(len(pred_res3), 1)
    a8 = np.array(pred_res_lstm)

    model2 = Sequential()
    model2.add(Embedding(input_dim=400, output_dim=128, input_length=7460, mask_zero=False))
    model2.add(Dropout(0.2))
    model2.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu'))
    model2.add(Activation(activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
    model2.add(Activation(activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(GlobalMaxPooling1D())
    model2.add(Dense(64))
    model2.add(Dropout(0.2))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=50, epochs=100, callbacks=callbacks, verbose=1)
    pred_res = model2.predict(x_test)
    pred_res2 = pred_res.flatten()
    pred_res_cnn = np.array(pred_res2).reshape(len(pred_res2), 1)
    a1 = np.array(pred_res_cnn)

    from catboost import CatBoostClassifier
    CAT = CatBoostClassifier(iterations=2983,
                             learning_rate=0.03823,
                             depth=8, subsample=0.7847,
                             rsm=0.8104,
                             task_type='CPU'
                             )

    CAT.fit(x_train, y_train)
    pred_res1 = CAT.predict_proba(x_test)[:, 1]
    pred_res_cat = np.array(pred_res1).reshape(len(pred_res1), 1)
    a2 = np.array(pred_res_cat)

    from sklearn import svm
    from sklearn.svm import SVC
    svc = svm.SVC(kernel='liblinear')  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    svc.fit(x_train, y_train)
    pred_res_svm = svc.predict_proba(x_test)[:, 1]
    pred_res_svm = np.array(pred_res_svm).reshape(len(pred_res_svm), 1)
    a3 = np.array(pred_res_svm)

    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=4906,
                        colsample_bytree=0.961263998009275,
                        learning_rate=0.004264475482427766,
                        max_depth=13,
                        min_child_weight=45.48826116566274,
                        subsample=0.8556594922541692)
    xgb.fit(x_train, y_train)
    pred_res_xgb = xgb.predict_proba(x_test)[:, 1]
    pred_res_xgb = np.array(pred_res_xgb).reshape(len(pred_res_xgb), 1)
    a4 = np.array(pred_res_xgb)

    from sklearn.ensemble import GradientBoostingClassifier
    gbdt = GradientBoostingClassifier(loss='deviance',
                                      n_estimators=2000,
                                      learning_rate=0.028673784262499755,
                                      max_depth=11,
                                      min_samples_leaf=59,
                                      min_samples_split=19,
                                      subsample= 0.6447933070462003,
                                      max_features=0.0001
                                      )
    gbdt.fit(x_train, y_train)
    pred_res_gbdt = gbdt.predict_proba(x_test)[:, 1]
    pred_res_gbdt = np.array(pred_res_gbdt).reshape(len(pred_res_gbdt), 1)
    a5 = np.array(pred_res_gbdt)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=3789,
                                max_depth=19,
                                min_samples_leaf=82,
                                min_samples_split=4,
                                bootstrap=True,
                                max_features='auto',
                                n_jobs=1,
                                random_state=3,
                                verbose=100,
                                oob_score=True
                                )

    rf.fit(x_train, y_train)
    pred_res_rf = rf.predict_proba(x_test)[:, 1]
    pred_res_rf = np.array(pred_res_rf).reshape(len(pred_res_rf), 1)
    a6 = np.array(pred_res_rf)


    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=3835,
                             learning_rate=0.0925472963334393,
                             max_depth=13,
                             cat_smooth=30.61541228449972,
                             bagging_fraction=0.6582040629836386,
                             bagging_freq=9,
                             feature_fraction=0.6887307635842295,
                             lambda_l1=0.9911958420007507,
                             lambda_l2 =34.290859455476834
                             )
    clf.fit(x_train, y_train,eval_set=(x_val, y_val), eval_metric="auc")
    pred_res_lgb = clf.predict_proba(x_test)[:, 1]
    pred_res_lgb = np.array(pred_res_lgb).reshape(len(pred_res_lgb), 1)
    a7 = np.array(pred_res_lgb)

    pred_res_sum = np.concatenate((a1, a3,a2, a4, a5, a6, a7,a8), axis=1)
    return np.array(pred_res_sum)

x_train=np.loadtxt('x_train.txt')
y_train=np.loadtxt('y_train.txt')
x_test=np.loadtxt('x_test.txt')
y_test=np.loadtxt('y_test.txt')
x_val=np.loadtxt('x_val.txt')
y_val=np.loadtxt('y_val.txt')
train=trainensemble(x_train,y_train,x_val, y_val, x_test,y_test)
test=testensemble(x_train,y_train,x_val, y_val, x_test, y_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=3789,
                            max_depth=19,
                            min_samples_leaf=82,
                            min_samples_split=4,
                            bootstrap=True,
                            max_features='auto',
                            n_jobs=1,
                            random_state=3,
                            verbose=100,
                            oob_score=True
                            )



rf.fit(train,y_train)
pred_res = rf.predict_proba(test)[:,1]
from sklearn import metrics
t = 0.5
pred_label = [1 if x > t else 0 for x in pred_res]
tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_test, y_pred=pred_label).ravel()
recall = metrics.recall_score(y_pred=pred_label, y_true=y_test)
precise = metrics.precision_score(y_pred=pred_label, y_true=y_test)

se = tp / (tp + fn)
sp = tn / (tn + fp)

acc = metrics.accuracy_score(y_pred=pred_label, y_true=y_test)
f1 = metrics.f1_score(y_pred=pred_label, y_true=y_test)
mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=y_test)

auc = metrics.roc_auc_score(y_true=y_test, y_score=pred_res)
ap = metrics.average_precision_score(y_score=pred_res, y_true=y_test)

print("tn", tn)
print("tp", tp)
print("fp", fp)
print("fn", fn)
print("se", se)
print("sp", sp)
print("precise", precise)
print("acc", acc)
print("f1", f1)
print("mcc", mcc)
print("auc", auc)
print("ap", ap)

