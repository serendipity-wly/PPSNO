import numpy as np
x_train=np.loadtxt('x_train.txt')
x_test=np.loadtxt('x_test.txt')
y_train=np.loadtxt('y_train.txt')
y_test=np.loadtxt('y_test.txt')
x_val=np.loadtxt('x_val.txt')
y_val=np.loadtxt('y_val.txt')

from keras.layers import Dropout, Embedding, Dense, Conv1D, GlobalMaxPooling1D, Activation
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

def CNN_model():
    model = Sequential()
    model.add(Embedding(input_dim=500, output_dim=128, input_length=7460, mask_zero=False))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',activation='relu'))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model=CNN_model()
model.fit(x_train, y_train, batch_size=128,validation_data=(x_val, y_val), epochs=100,callbacks=callbacks)
model.evaluate(x_test, y_test)
pred_res = model.predict(x_test)
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

