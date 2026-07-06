import cv2
import os
import threading
import time
from ultralytics import YOLO

# --- 1. SETUP ---
model = YOLO('yolo11n.pt')
model.to('mps') 

# --- 2. TRACKING VARIABLES ---
system_greeting_done = False
last_speech_time = 0
empty_room_start_time = 0
INTERVAL = 60         
RESET_DELAY = 15      

def speak(text):
    os.system(f"say -v Amelie '[[volm 1.0]] {text}'")

# --- 3. MAIN LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, classes=[0], conf=0.5, verbose=False, device="mps")
    current_count = len(results[0].boxes)
    now = time.time()
    
    # --- SPEECH LOGIC ---
    if current_count > 0:
        empty_room_start_time = 0 
        count_text = "une personne" if current_count == 1 else f"{current_count} personnes"
        
        if not system_greeting_done:
            full_msg = f"Bonjour, Bienvenue en tram de Montpellier. Il y a actuellement {count_text} à bord."
            threading.Thread(target=speak, args=(full_msg,), daemon=True).start()
            system_greeting_done = True
            last_speech_time = now 

        elif now - last_speech_time >= INTERVAL:
            update_msg = f"Mise à jour : Il y a {count_text} à bord."
            threading.Thread(target=speak, args=(update_msg,), daemon=True).start()
            last_speech_time = now 
    else:
        if empty_room_start_time == 0:
            empty_room_start_time = now
        if now - empty_room_start_time > RESET_DELAY:
            system_greeting_done = False

    # --- 4. VISUAL OUTPUT (BIG & BOLD) ---
    annotated_frame = results[0].plot()

    # 1. Draw a dark semi-transparent rectangle for a professional look
    # (Top-left corner, Bottom-right corner, Color BGR, -1 means filled)
    cv2.rectangle(annotated_frame, (0, 0), (450, 80), (0, 0, 0), -1)

    # 2. Add the BIG BOLD text
    # Parameters: (image, text, position, font, scale, color, thickness)
    cv2.putText(
        annotated_frame, 
        f"PASSAGERS: {current_count}", 
        (20, 55),                   # Position (x, y)
        cv2.FONT_HERSHEY_DUPLEX,    # A more professional "bold" font
        1.8,                        # Font Scale (Size)
        (0, 255, 0),                # Color (Green)
        4                           # Thickness (Boldness)
    )

    cv2.imshow("Tram Audio Monitor", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()