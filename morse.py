import numpy as np
import cv2
import mediapipe as mp
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list
# 根據手指角度的串列內容，返回對應的手勢名稱
def hand_pos(finger_angle):
    #global last_time
    #current_time = time.time()
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度`
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度
    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1<50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:#短
        #if current_time - last_time > 100:
            #last_time = current_time
        #time.sleep(1)
        return 0
    elif f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:#長
       # if current_time - last_time > 100:
           # last_time = current_time
        #time.sleep(1)
        return 1
    elif f1 < 50 and f2 <50 and f3<50 and f4<50 and f5<50:#手指全部伸直
        return 2

def show(text, morse_str):
    image = np.zeros((480, 640, 3), np.uint8)
    # Fill image with gray color(set each pixel to gray)
    image[:] = (255, 255, 255)
    cv2.putText(image, morse_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
      1, (0, 0, 0), 1, cv2.LINE_AA)
    
    morse = "morse: "+ text
    cv2.putText(image, morse, (300, 50), cv2.FONT_HERSHEY_SIMPLEX,
      1, (0, 0, 0), 1   , cv2.LINE_AA)
    cv2.imshow('Result', image)
        
cap = cv2.VideoCapture(0)            # 讀取攝影機
fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
lineType = cv2.LINE_AA               # 印出文字的邊框
morse_code_dict = {
    '01': 'A', '1000': 'B', '1010': 'C', '100': 'D', '0': 'E',
    '0010': 'F', '110': 'G', '0000': 'H', '00': 'I', '0111': 'J',
    '101': 'K', '0100': 'L', '11': 'M', '10': 'N', '111': 'O',
    '0110': 'P', '1101': 'Q', '010': 'R', '000': 'S', '1': 'T',
    '001': 'U', '0001': 'V', '011': 'W', '1001': 'X', '1011': 'Y',
    '1100': 'Z', '11111': '0', '01111': '1', '00111': '2', '00011': '3',
    '00001': '4', '00000': '5', '10000': '6', '11000': '7', '11100': '8',
    '11110': '9'
}
last_time = time.time()
morse = []  #morse表示
morse_str = ""
text = ""
# mediapipe 啟用偵測手掌
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    w, h = 540, 310                                  # 影像尺寸
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (w,h))                 # 縮小尺寸，加快處理效率
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB 色彩
        current_time = time.time()
        if current_time - last_time > 1.5:#每一秒偵測一次
            last_time = current_time
            results = hands.process(img2)                # 偵測手勢
            if results.multi_hand_landmarks:
                print("strat detect")
                for hand_landmarks in results.multi_hand_landmarks:
                    finger_points = []                   # 記錄手指節點座標的串列
                    for i in hand_landmarks.landmark: # 將 21 個節點換算成座標，記錄到 finger_points
                        x = i.x*w
                        y = i.y*h
                        finger_points.append((x,y))
                    if finger_points:
                        finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
                        text = hand_pos(finger_angle)   # 取得手勢所回傳的內容
                        print("text",text)
                        if text != 2 and text != None:#手勢判斷加入list
                            morse.append(text)
                            show('', str(text))
                        else:
                            morse_str = ''.join(str(i) for i in morse)
                            print("morse_str", morse_str) 
                            if morse_str in morse_code_dict:#輸出判斷
                                output = morse_code_dict[morse_str]
                                print("output", output)
                                show(output, morse_str)
                            else:
                                show("None",'')
                            morse.clear()
                    
        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
