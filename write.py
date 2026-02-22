import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# ----------------------------
# Model ve MediaPipe Tasks setup
# ----------------------------
MODEL_PATH = "hand_landmarker.task"  # El modeli dosya yolu

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# ----------------------------
# Kamera ve Canvas
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas = None
prev_x, prev_y = None, None

colors = {
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255)
}
color_names = list(colors.keys())
current_color_idx = 0
current_color = colors[color_names[current_color_idx]]
erasing = False
frame_count = 0
brush_size = 5  # default

# ----------------------------
# Yardımcı fonksiyonlar
# ----------------------------
def draw_ui(frame):
    for i, color in enumerate(colors.values()):
        cv2.rectangle(frame, (50+i*60, 10), (90+i*60, 50), color, -1)
    cv2.rectangle(frame, (400, 10), (480, 50), (50,50,50), -1)
    cv2.putText(frame, "ERASE", (405,40), cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)
    cv2.rectangle(frame, (500,10),(580,50),(50,50,50),-1)
    cv2.putText(frame,"CLEAR",(505,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)

def fingers_up(hand_landmarks):
    tips = [4,8,12,16,20]
    fingers=[]
    # Baş parmak
    if hand_landmarks[tips[0]].x < hand_landmarks[tips[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Diğer parmaklar
    for tip in tips[1:]:
        if hand_landmarks[tip].y < hand_landmarks[tip-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def get_finger_pos(hand_landmarks, tip=8, shape=(1280,720)):
    w,h = shape
    lm = hand_landmarks[tip]
    x = int(lm.x * w)
    y = int(lm.y * h)
    return x,y

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# ----------------------------
# Ana döngü
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros((h,w,3),dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # MediaPipe Tasks API
    result = landmarker.detect_for_video(mp_image, frame_count)
    frame_count += 1
    mode = "Move"

    draw_ui(frame)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Landmarkları göster
            for lm in hand_landmarks:
                lx, ly = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame,(lx,ly),5,(0,255,0),-1)

            fingers = fingers_up(hand_landmarks)
            ix, iy = get_finger_pos(hand_landmarks, 8, (w,h))   # İşaret parmağı
            tx, ty = get_finger_pos(hand_landmarks, 4, (w,h))   # Baş parmak
            mx, my = get_finger_pos(hand_landmarks, 12, (w,h))  # Orta parmak

            # Fırça boyutu → sadece ilk ayarlamada işaret + baş parmak mesafesi
            if prev_x is None and prev_y is None:
                brush_size = max(2, int(distance((ix,iy),(tx,ty))//2))

            # Renk seçimi
            for i, color in enumerate(colors.values()):
                if 50+i*60 < ix < 90+i*60 and 10 < iy < 50:
                    current_color_idx = i
                    current_color = color

            # Silgi modu → işaret + orta + yüzük parmak
            erasing = fingers[1]==1 and fingers[2]==1 and fingers[3]==1

            # Canvas temizleme
            if 500<ix<580 and 10<iy<50 and fingers[1]==1:
                canvas = np.zeros((h,w,3),dtype=np.uint8)

            # Çizim → sadece işaret parmağı
            if fingers[1]==1 and fingers[2]==0:
                mode="Draw"
                if prev_x is not None and prev_y is not None:
                    color_to_draw = (0,0,0) if erasing else current_color
                    cv2.line(canvas,(prev_x,prev_y),(ix,iy),color_to_draw,brush_size)
                prev_x, prev_y = ix, iy

            # Taşıma → işaret + orta parmak
            elif fingers[1]==1 and fingers[2]==1:
                mode="Move"
                if prev_x is not None and prev_y is not None:
                    dx, dy = ix-prev_x, iy-prev_y
                    canvas = np.roll(canvas, dx, axis=1)
                    canvas = np.roll(canvas, dy, axis=0)
                prev_x, prev_y = ix, iy
            else:
                prev_x, prev_y = None, None

            cv2.circle(frame,(ix,iy),brush_size,current_color,-1)

    # Canvas + frame birleştirme
    mask = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    output = cv2.add(frame_bg, canvas_fg)

    cv2.putText(output, f"Mode: {mode} Brush: {brush_size}", (10,h-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv2.imshow("Finger Drawing 2D", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()