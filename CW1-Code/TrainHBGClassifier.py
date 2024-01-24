import os
import pickle
import time

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
quickest_time = 1000000
lowest_score = 100000
for i in range(10):
    start_time = time.time()

    loadedData = pickle.load(open('data.pickle', 'rb'))

    data = np.asarray(loadedData['data'])
    labels = np.asarray(loadedData['labels'])



    xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.7, shuffle=True, stratify=labels)

    model = HistGradientBoostingClassifier(max_iter=100).fit(xTrain, yTrain)

    yPredict = model.score(xTest, yTest)


    print(f"{yPredict * 100}% of samples were classified correctly!")
    print(f"Trained in: {time.time() - start_time}")
    f = open('model.p', 'wb')
    pickle.dump({'model': model}, f)
    f.close()