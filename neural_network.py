import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ##########################

start = time.time()

np.random.seed(8)

pd.read_csv('heart_disease.csv')
hdData = pd.DataFrame(hd_file)

X = hdData.drop('target', axis = 1).astype(float)
Y = hdData['target']

X_train, X_test, Y_train, Y_test = train_test_split(
 X, Y, test_size=0.1, random_state=5)

def create_neural_network():
    model = Sequential()
    model.add(Dense(26, input_dim=13, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_neural_network, epochs=12, batch_size=1, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
scores = []
for train_idx, test_idx in kfold.split(X_train, Y_train):
    pipeline.fit(X_train.iloc[train_idx], Y_train.iloc[train_idx])
    scores.append(pipeline.score(X_train.iloc[test_idx], Y_train.iloc[test_idx]))
    print(scores[-1])
print("mean (std): %.2f%% (%.2f%%)" % (np.mean(scores)*100, np.std(scores)*100)) 
plt.plot(scores)
real_results = pipeline.score(X_test, Y_test)
print("real test" + str(real_results))

print(time.time() - start)
