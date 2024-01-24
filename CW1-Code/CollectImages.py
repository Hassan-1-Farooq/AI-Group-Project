import os
import cv2
from CONFIG import *

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


totalHandSigns = 4 # change this number if we are capturing more signs
datasetSize = 1000

capture = cv2.VideoCapture(0)

for i in range(totalHandSigns):
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.mkdir(os.path.join(DATA_DIR, str(i)))

    print(f"Collecting data for class {i}")

    done = False
    while True:
        result, frame = capture.read()
        cv2.putText(frame, "Ready? Press 'Q'!",
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


        cv2.imshow('frame', frame)
        if cv2.waitKey(WAIT_KEY) == QUIT_KEY:
            break

    counter = 0
    while counter < datasetSize:
        result, frame = capture.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(WAIT_KEY)
        cv2.imwrite(os.path.join(DATA_DIR, str(i), f"{counter}.jpg"), frame)

        counter += 1

capture.release()
cv2.destroyAllWindows()
