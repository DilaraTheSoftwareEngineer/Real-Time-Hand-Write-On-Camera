import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import os

# ----------------------------
# MediaPipe Setup
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
# Ayarlar ve Global Değişkenler
# ----------------------------
cap = cv2.VideoCapture(0)
W, H = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

# Çizim verileri: (x1, y1, x2, y2, renk, kalınlık, silgi_mi)
all_lines = [] 

# Kamera/Bakış açısı değişkenleri
offset_x, offset_y = 0, 0
zoom_level = 1.0

prev_x, prev_y = None, None
prev_dist = None  
brush_size = 5
current_color = (255, 0, 255)
frame_count = 0
last_save_time = 0
show_palette = False

# Renk Skalası
color_picker_img = np.zeros((50, 300, 3), dtype=np.uint8)
for i in range(300):
    hue = int(180 * i / 300)
    color_picker_img[:, i] = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]

def draw_ui(frame):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
    for i, col in enumerate(colors):
        cv2.rectangle(frame, (50 + i*70, 10), (110 + i*70, 70), col, -1)
    cv2.rectangle(frame, (400, 10), (500, 70), (100, 0, 100), -1)
    cv2.putText(frame, "PALET", (415, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(frame, (520, 10), (620, 70), (50, 50, 50), -1)
    cv2.putText(frame, "CLR", (545, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(frame, (1100, 10), (1160, 70), (100, 100, 100), -1)
    cv2.putText(frame, "+", (1115, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.rectangle(frame, (1180, 10), (1240, 70), (100, 100, 100), -1)
    cv2.putText(frame, "-", (1200, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    if show_palette:
        frame[80:130, 400:700] = color_picker_img

def get_fingers_state(hand_landmarks):
    fingers = []
    if hand_landmarks[4].x < hand_landmarks[3].x: fingers.append(1)
    else: fingers.append(0)
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
    
    mode = "Secim"
    draw_ui(frame)
    temp_canvas = np.zeros((H, W, 3), dtype=np.uint8)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        fingers = get_fingers_state(landmarks)
        tx, ty = int(landmarks[4].x * W), int(landmarks[4].y * H)
        ix, iy = int(landmarks[8].x * W), int(landmarks[8].y * H)

        for lm in landmarks:
            cv2.circle(frame, (int(lm.x*W), int(lm.y*H)), 5, (0, 255, 0), -1)

        # 1. KAYDET (👍 - Sadece Baş Parmak)
        if fingers == [0, 0, 0, 0, 1]:
            mode = "PNG KAYDEDILIYOR"
            if time.time() - last_save_time > 3:
                desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                save_canvas = np.zeros((H, W, 4), dtype=np.uint8)
                for line in all_lines:
                    color = line[4] if not line[6] else (0,0,0)
                    cv2.line(save_canvas, (line[0], line[1]), (line[2], line[3]), (*color, 255), line[5])
                cv2.imwrite(os.path.join(desktop, f"cizim_{int(time.time())}.png"), save_canvas)
                last_save_time = time.time()

        # 2. MERKEZ ODAKLI ZOOM (🤌 - Baş + İşaret)
        elif fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
            mode = "Zoom"
            dist = math.hypot(ix - tx, iy - ty)
            mid_x, mid_y = (ix + tx) // 2, (iy + ty) // 2
            
            if prev_dist is not None:
                # Önceki zoom değerini sakla
                old_zoom = zoom_level
                zoom_change = 1.05 if dist > prev_dist + 10 else (0.95 if dist < prev_dist - 10 else 1.0)
                zoom_level = max(0.1, min(10.0, zoom_level * zoom_change))
                
                # Zoom yapılırken bakılan noktanın kaymaması için offset'i güncelle
                if zoom_change != 1.0:
                    offset_x = mid_x - (mid_x - offset_x) * (zoom_level / old_zoom)
                    offset_y = mid_y - (mid_y - offset_y) * (zoom_level / old_zoom)
                    
            prev_dist = dist
            prev_x, prev_y = None, None

        # 3. SİLGİ (✌️ - İşaret + Orta)
        elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:
            mode = "Silgi"
            v_ix = int((ix - offset_x) / zoom_level)
            v_iy = int((iy - offset_y) / zoom_level)
            if prev_x is not None:
                all_lines.append((prev_x, prev_y, v_ix, v_iy, (0,0,0), int((brush_size+20)/zoom_level), True))
            prev_x, prev_y = v_ix, v_iy

        # 4. ÇİZİM VE MENÜ (☝️ - İşaret)
        elif fingers[1] == 1 and sum(fingers) == 1:
            if iy < 80:
                if 50 < ix < 400: current_color = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,255)][(ix-50)//70]
                elif 400 < ix < 500: show_palette = not show_palette
                elif 520 < ix < 620: all_lines = []
                elif 1100 < ix < 1160: brush_size += 2
                elif 1180 < ix < 1240: brush_size = max(1, brush_size - 2)
            
            if show_palette and 80 < iy < 130 and 400 < ix < 700:
                current_color = tuple(map(int, color_picker_img[0, ix-400]))

            mode = "Cizim"
            v_ix = int((ix - offset_x) / zoom_level)
            v_iy = int((iy - offset_y) / zoom_level)
            if iy > 80:
                if prev_x is not None:
                    all_lines.append((prev_x, prev_y, v_ix, v_iy, current_color, int(brush_size/zoom_level), False))
                prev_x, prev_y = v_ix, v_iy
            else: prev_x, prev_y = None, None

        # 5. TAŞIMA (✋ - Tam El)
        elif sum(fingers) >= 4:
            mode = "Tasima"
            if prev_x is not None:
                offset_x += (ix - prev_x)
                offset_y += (iy - prev_y)
            prev_x, prev_y = ix, iy
        else:
            prev_x, prev_y = None, None
            prev_dist = None

    # --- ÇİZİMLERİ RENDER ET ---
    for line in all_lines:
        x1, y1 = int(line[0] * zoom_level + offset_x), int(line[1] * zoom_level + offset_y)
        x2, y2 = int(line[2] * zoom_level + offset_x), int(line[3] * zoom_level + offset_y)
        thick = max(1, int(line[5] * zoom_level))
        
        # Sadece ekran içindeyse çiz
        if -100 < x1 < W+100 or -100 < x2 < W+100:
            cv2.line(temp_canvas, (x1, y1), (x2, y2), line[4], thick)

    # UI Önizleme
    cv2.circle(frame, (W-100, H-100), brush_size, current_color, -1)
    cv2.putText(frame, f"Size: {brush_size}", (W-150, H-50), 1, 1, (255, 255, 255), 2)

    # Birleştir
    gray = cv2.cvtColor(temp_canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    output = cv2.add(frame, temp_canvas)

    cv2.putText(output, f"MOD: {mode} | Zoom: {zoom_level:.2f}", (10, H-20), 1, 1.5, (0, 255, 0), 2)
    cv2.imshow("Smart Canvas Pro", output)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()