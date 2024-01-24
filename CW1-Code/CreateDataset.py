import os
import pickle
import mediapipe as mp
import cv2

from CONFIG import DATA_DIR

mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
mpStyles = mp.solutions.drawing_styles

hands = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

for directory in os.listdir(DATA_DIR):
    for imagePath in os.listdir(os.path.join(DATA_DIR, directory)):
        dataAUX = []
        handX = []
        handY = []

        image = cv2.imread(os.path.join(DATA_DIR, directory, imagePath))
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(imageRGB)
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                for i in range(len(handLandmarks.landmark)):
                    landX = handLandmarks.landmark[i].x
                    landY = handLandmarks.landmark[i].y

                    handX.append(landX)
                    handY.append(landY)

                    dataAUX.append(landX - min(handX))
                    dataAUX.append(landY - min(handY))

            data.append(dataAUX)
            labels.append(directory)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
