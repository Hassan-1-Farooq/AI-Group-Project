import pickle
import cv2
import mediapipe as mp
import numpy as np
from CONFIG import *

models = pickle.load(open('./model.p', 'rb'))
model = models['model']

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
mpStyle = mp.solutions.drawing_styles

hands = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}  # add all the sign labels here corresponding to data subdirectories
while True:

    dataAUX = []
    handX = []
    handY = []

    result, frame = capture.read()

    H, W, _ = frame.shape

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frameRGB)
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mpDrawing.draw_landmarks(
                frame,
                handLandmarks,
                mpHands.HAND_CONNECTIONS,
                mpStyle.get_default_hand_landmarks_style(),
                mpStyle.get_default_hand_connections_style())

            for i in range(len(handLandmarks.landmark)):
                landX = handLandmarks.landmark[i].x
                landY = handLandmarks.landmark[i].y

                handX.append(landX)
                handY.append(landY)
                dataAUX.append(landX - min(handX))
                dataAUX.append(landY - min(handY))

        x1 = int(min(handX) * W) - 10
        y1 = int(min(handY) * H) - 10
        x2 = int(max(handX) * W) - 10
        y2 = int(max(handY) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        # TODO: Try using the method 'predict_proba' instead of 'predict'.
        #  return a probability vector for all the classes.
        #  Taking the largest number will give confidence value.
        prediction = model.predict([np.asarray(dataAUX)])
        predictedCharacter = labels[int(prediction[0])]

        cv2.putText(frame, predictedCharacter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0),
                    FONT_THICKNESS, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
