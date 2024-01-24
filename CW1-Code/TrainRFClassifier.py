import pickle
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

quickest_time = 1000000
lowest_score = 100000
for i in range(10):

    start_time = time.time()
    loadedData = pickle.load(open('./data.pickle', 'rb'))

    data = np.asarray(loadedData['data'])
    labels = np.asarray(loadedData['labels'])

    xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()

    model.fit(xTrain, yTrain)

    yPredict = model.predict(xTest)

    score = accuracy_score(yPredict, yTest)
    if score * 100 < lowest_score:
        lowest_score = score*100
    time_to_run = time.time() - start_time
    if time_to_run < quickest_time:
        quickest_time = time_to_run
    f = open('model.p', 'wb')
    pickle.dump({'model': model}, f)
    f.close()
print(f"Score: {lowest_score}%")
print(f"Trained in: {quickest_time}")
