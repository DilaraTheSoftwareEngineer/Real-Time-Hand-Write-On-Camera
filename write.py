import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# ----------------------------
# Model ve MediaPipe Tasks setup
# ----------------------------
MODEL_PATH = "hand_landmarker.task"

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
W, H = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

canvas = np.zeros((H, W, 3), dtype=np.uint8)
prev_x, prev_y = None, None
prev_dist = None
brush_size = 5
current_color = (255, 0, 255)
frame_count = 0

def draw_ui(frame):
    # Renk Paleti
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
    for i, col in enumerate(colors):
        cv2.rectangle(frame, (50 + i*70, 10), (110 + i*70, 70), col, -1)
    
    # CLEAR Butonu
    cv2.rectangle(frame, (450, 10), (550, 70), (50, 50, 50), -1)
    cv2.putText(frame, "CLR", (465, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Brush Size Butonları
    cv2.rectangle(frame, (1100, 10), (1160, 70), (100, 100, 100), -1)
    cv2.putText(frame, "+", (1115, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.rectangle(frame, (1180, 10), (1240, 70), (100, 100, 100), -1)
    cv2.putText(frame, "-", (1200, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

def get_fingers_state(hand_landmarks):
    # Parmakların açık olup olmadığını kontrol eder
    # [Baş, İşaret, Orta, Yüzük, Serçe]
    fingers = []
    # Baş parmak (Uç nokta ile eklem arası mesafe/pozisyon)
    if hand_landmarks[4].x < hand_landmarks[3].x: fingers.append(1)
    else: fingers.append(0)
    # Diğerleri
    for tip in [8, 12, 16, 20]:
        if hand_landmarks[tip].y < hand_landmarks[tip-2].y: fingers.append(1)
        else: fingers.append(0)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, frame_count)
    frame_count += 1
    
    mode = "Selection"
    draw_ui(frame)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        fingers = get_fingers_state(landmarks)
        
        # Koordinatlar
        ix, iy = int(landmarks[8].x * W), int(landmarks[8].y * H)   # İşaret
        mx, my = int(landmarks[12].x * W), int(landmarks[12].y * H) # Orta
        
        # --- EL NOKTALARINI ÇİZ --- (Solutions kullanmadan manuel çizim)
        for lm in landmarks:
            cx, cy = int(lm.x * W), int(lm.y * H)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # --- MOD KONTROLLERİ ---
        
        # 1. FIRÇA BOYUTU VE RENK (Sadece İşaret Parmağı)
        if fingers[1] == 1 and fingers[2] == 0:
            if iy < 80:
                if 50 < ix < 400: # Renk seçimi
                    idx = (ix - 50) // 70
                    colors_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
                    if idx < len(colors_list): current_color = colors_list[idx]
                elif 450 < ix < 550: # Clear
                    canvas = np.zeros((H, W, 3), dtype=np.uint8)
                elif 1100 < ix < 1160: # Boyut +
                    brush_size = min(brush_size + 1, 50)
                elif 1180 < ix < 1240: # Boyut -
                    brush_size = max(brush_size - 1, 1)
            
            mode = "Drawing"
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), current_color, brush_size)
            prev_x, prev_y = ix, iy

        # 2. SİLGİ (İşaret ve Orta Parmak Havada)
        elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:
            mode = "Erasing"
            cv2.circle(frame, (ix, iy), brush_size + 20, (255, 255, 255), 2)
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 0), brush_size + 30)
            prev_x, prev_y = ix, iy

        # 3. TAŞIMA (Tüm El Açık)
        elif sum(fingers) >= 4:
            mode = "Moving"
            if prev_x is not None:
                dx, dy = ix - prev_x, iy - prev_y
                # np.roll sayesinde taşırken görüntü kaybolmaz, diğer taraftan geri gelir.
                canvas = np.roll(canvas, dx, axis=1)
                canvas = np.roll(canvas, dy, axis=0)
            prev_x, prev_y = ix, iy

        # 4. ZOOM (İşaret + Orta + Baş Parmak)
        elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1:
            mode = "Zooming"
            dist = math.hypot(ix - mx, iy - my)
            if prev_dist is not None:
                scale = 1.03 if dist > prev_dist + 5 else (0.97 if dist < prev_dist - 5 else 1.0)
                if scale != 1.0:
                    M = cv2.getRotationMatrix2D((ix, iy), 0, scale)
                    canvas = cv2.warpAffine(canvas, M, (W, H))
            prev_dist = dist
        else:
            prev_dist = None
            prev_x, prev_y = None, None

    # Görüntüleri Birleştir
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final_output = cv2.add(frame_bg, canvas)

    cv2.putText(final_output, f"Mode: {mode} Size: {brush_size}", (10, H-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Tasks Hand Landmarker", final_output)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()