import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque


model = load_model("hand_sign_model.h5")
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y']


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


sentence = ""
last_letter = ""
letter_buffer = deque(maxlen=15)

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    for i in range(1, 5): 
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers) == 4 


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) 
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            x_min, y_min = max(0, x_min - 20), max(0, y_min - 20)
            x_max, y_max = min(w, x_max + 20), min(h, y_max + 20)
            roi = frame[y_min:y_max, x_min:x_max]

            try:
                resized_roi = cv2.resize(roi, (64, 64))
                norm_roi = resized_roi / 255.0
                input_array = np.expand_dims(norm_roi, axis=0)
                predictions = model.predict(input_array, verbose=0)
                predicted_class = class_labels[np.argmax(predictions)]
                confidence = np.max(predictions)

                if confidence > 0.8:
                    letter_buffer.append(predicted_class)
                    most_common = max(set(letter_buffer), key=letter_buffer.count)

                    if letter_buffer.count(most_common) > 10 and (most_common != last_letter or letter_buffer.count(most_common) == 15):
                        if fingers_up(hand_landmarks):
                            sentence += " "
                        else:
                            sentence += most_common
                        last_letter = most_common
                        letter_buffer.clear()

                cv2.rectangle(frame, (x_min, y_min - 60), (x_max, y_min - 10), (0, 255, 0), -1)
                cv2.putText(frame, f"{predicted_class}", (x_min + 10, y_min - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

            except Exception as e:
                print("Error:", e)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.rectangle(frame, (10, 400), (630, 460), (50, 50, 50), -1)
    cv2.putText(frame, sentence, (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow("Smart Hand Sign Recognizer", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        break
    elif key == ord('c'): 
        sentence = ""
        last_letter = ""
        letter_buffer.clear()

cap.release()
cv2.destroyAllWindows()
