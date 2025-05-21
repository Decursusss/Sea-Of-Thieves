import cv2
import numpy as np
import threading
import win32api
import win32con
import time
import pydirectinput
import keyboard
from ultralytics import YOLO
from wayPoints import ActionRecorder
from pynput.keyboard import Key

def move_mouse(dx: int, dy: int):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,
                         int(dx), int(dy), 0, 0)

def press_left_mouse_button():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def realese_left_moust_button():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def test_mouse_movement(direction):
    if direction == "right":
        move_mouse(100, -20)
    elif direction == "left":
        move_mouse(-100, -20)
    elif direction == "down":
        move_mouse(0, 100)


def toggle_fishing():
    global fishing_active
    fishing_active = not fishing_active
    print("Авто-рыбалка ВКЛ 🟢" if fishing_active else "Авто-рыбалка ВЫКЛ 🔴")

def minigame(frame, bobber_pos):
    global prev_frame, holding_direction, CENTER_TOLERANCE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        horizontal_flow = flow[..., 0]
        mean_flow = np.mean(horizontal_flow)

        vertical_flow = flow[..., 1]
        mean_vertical_flow = np.mean(vertical_flow)

        if mean_flow > 0.56:
            holding_direction = "right"
        elif mean_flow < -0.56:
            holding_direction = "left"

        frame_w = frame.shape[1]
        if bobber_pos is not None:
            center_x = frame_w / 2
            if abs(bobber_pos[0] - center_x) <= frame_w * CENTER_TOLERANCE:
                holding_direction = "down"

        test_mouse_movement(holding_direction)

        print(f"Вертикальный поток: {mean_vertical_flow:.2f} | Горизонтальный поток: {mean_flow:.2f} | Действие: {holding_direction}")

    prev_frame = gray.copy()

def find_bobber(frame):
    results = model(frame, conf=0.15, verbose=False)
    bobber_position = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            class_name = model.names[class_id] if class_id in model.names else f"ID {class_id}"

            if class_name.lower() == "bloba":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                bobber_position = ((x1 + x2) // 2, (y1 + y2) // 2)

    return bobber_position, frame

def find_finisher(frame, templates, threshold=0.8):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for idx, template in enumerate(templates):
        if template is None:
            continue

        h, w = template.shape[:2]
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            return True
    return False

def aditional_finisher(frame, template, threshold=0.8):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = template.shape[:2]
    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        return True
    return False

def press_key(key_code, hold_time=0.1):
    win32api.keybd_event(key_code, 0, 0, 0)
    time.sleep(hold_time)
    win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)

def hold_key(key_code, hold_time):
    win32api.keybd_event(key_code, 0, 0, 0)
    time.sleep(hold_time)
    win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)

def restart_game():
    global prev_frame, finisher, holding_direction, counter, finisher_counter, minigame_starter, minigame_starter_counter, vender_counter
    prev_frame = None
    finisher = False
    minigame_starter = False
    holding_direction = ""
    counter = 0
    finisher_counter = 0
    minigame_starter_counter = 0
    vender_counter = 0

    hold_key(ord('F'), 5)

    win32api.keybd_event(ord('Q'), 0, 0, 0)
    time.sleep(1)

    press_key(ord('F'))
    time.sleep(1)

    press_key(ord('5'))

    win32api.keybd_event(ord('Q'), 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(1)

    press_left_mouse_button()
    time.sleep(2)
    realese_left_moust_button()

TEMPLATE_PATHS = [
    "templates/templ1(FINISHER).png",
    "templates/templ2(FINISHER).png",
]

template_names = [path.split('/')[-1] for path in TEMPLATE_PATHS]
templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in TEMPLATE_PATHS]

template2 = cv2.imread("templates/aditionFinisher2.png", cv2.IMREAD_GRAYSCALE)


cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FPS, 40)

model = YOLO("runs/detect/train/weights/best.pt")

CENTER_TOLERANCE = 0.10
prev_frame = None
finisher = False
holding_direction = ""

minigame_starter = False
fishing_active = False
first_time = True
keyboard.add_hotkey("F4", toggle_fishing)

counter = 0
finisher_counter = 0
minigame_starter_counter = 0
vender_counter = 0

recorder = ActionRecorder()

print("Добро пожаловать в автоматическую рыбалку для SOT")
print("*Для начала запишите маршрут до торговца по нажатию клавиши F1 для начала записи и F2 для остановки записи.")
print("*После записи маршрута проверьте его по нажатию F3")
print("*Когда все приготовление готовы нажмите F4 для запуска автоматической рыбалки")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()
    new_frame = frame_copy

    # test = aditional_finisher(frame_copy, template2, threshold=0.81)
    # print(test)

    if fishing_active:
        if first_time:
            restart_game()
            first_time = False

        position, new_frame = find_bobber(frame_copy)

        if position != None and minigame_starter == False:
            minigame_starter_counter += 1
            if minigame_starter_counter >= 7:
                minigame_starter = True

        if minigame_starter:
            minigame(frame_copy, position)
            finisher = find_finisher(frame_copy, templates, threshold=0.9)

            if position == None:
                counter += 1
            else:
                counter = 0
                realese_left_moust_button()

            if finisher:
                finisher_counter += 1
            else:
                finisher_counter = 0

            if counter >= 5:
                press_left_mouse_button()
                finisher = find_finisher(frame_copy, templates, threshold=0.9)

            if finisher_counter >= 5:
                if vender_counter >= 5:
                    vender_counter = 0
                    replay_thread = recorder.on_execute(Key.f3)

                    if replay_thread:
                        replay_thread.join()

                restart_game()
                vender_counter += 1


    cv2.imshow('Fishing Mini-Game', new_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()