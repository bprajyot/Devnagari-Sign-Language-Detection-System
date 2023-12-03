import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'अ',
    1: 'आ',
    2: 'इ',
    3: 'ई',
    4: 'उ',
    5: 'ऊ',
    6: 'ए',
    7: 'ऐ',
    8: 'ओ',
    9: 'औ',
    10: 'क',
    11: 'ख',
    12: 'ग',
    13: 'घ',
    14: 'च',
    15: 'छ',
    16: 'ज',
    17: 'झ',
    18: 'ट',
    19: 'ठ',
    20: 'ड',
    21: 'ढ',
    22: 'ण',
    23: 'त',
    24: 'थ',
    25: 'द',
    26: 'ध',
    27: 'न',
    28: 'प',
    29: 'फ',
    30: 'ब',
    31: 'भ',
    32: 'म',
    33: 'य',
    34: 'र',
    35: 'ल',
    36: 'व',
    37: 'श',
    38: 'स',
    39: 'ह',
    40: 'ळ',
    41: 'क्ष',
    42: 'ज्ञ',
}


font_path = r"C:\Users\bpraj\Desktop\Noto_Sans_Devanagari\static\NotoSansDevanagari-Regular.ttf"

font_size = 25
font = ImageFont.truetype(font_path, font_size)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks,  
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        data_aux = data_aux[:42]

        max_sequence_length = 3900
        data_aux = np.pad(np.asarray(data_aux), (0, max_sequence_length - len(data_aux)), 'constant')[:max_sequence_length]

        data_aux_reshaped = np.reshape(data_aux, (1, -1))

        prediction = model.predict(data_aux_reshaped)
        predicted_character = labels_dict[int(prediction[0])]
        print(predicted_character)

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)
        draw.text((x1, y1 - font_size - 10), predicted_character, font=font, fill=(0, 0, 0))

        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
