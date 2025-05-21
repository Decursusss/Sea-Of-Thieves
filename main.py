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

def press_righ_mouse_button():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

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
    frame_w = frame.shape[1]
    center_x = frame_w / 2

    if prev_frame is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mean_flow = np.mean(flow[..., 0])
        mean_vertical_flow = np.mean(flow[..., 1])

        if bobber_pos is not None:
            offset = bobber_pos[0] - center_x
            if abs(offset) <= frame_w * CENTER_TOLERANCE:
                holding_direction = "down"
            elif offset < 0:
                holding_direction = "right"
            else:
                holding_direction = "left"
        else:
            if mean_flow > 0.56:
                holding_direction = "right"
            elif mean_flow < -0.56:
                holding_direction = "left"

        test_mouse_movement(holding_direction)

        print(f"Вертикальный поток: {mean_vertical_flow:.2f} | "
              f"Горизонтальный поток: {mean_flow:.2f} | "
              f"Действие: {holding_direction}")

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

def is_fishing_finished(frame,template, threshold=0.8):
    h, w, _ = frame.shape

    roi_w, roi_h = w // 6, h // 6
    x1 = 0
    y1 = 0
    x2 = max(300, template.shape[1]) + 20
    y2 = max(120, template.shape[0]) + 20

    # x1 = w - roi_w - 110
    # y1 = 90
    # x2 = w + 100
    # y2 = roi_h + 200

    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Fishing ROI", roi)

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray_roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print(f"Template matching max_val: {max_val}")

    if max_val >= threshold:
        return True
    return False


    # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #
    # lower_white = np.array([0, 0, 200])
    # upper_white = np.array([180, 40, 255])
    #
    # white_mask = cv2.inRange(hsv, lower_white, upper_white)
    #
    # lower_gold = np.array([20, 100, 150])
    # upper_gold = np.array([35, 255, 255])
    #
    # gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
    #
    # cv2.imshow("Red Mask", white_mask)
    #
    # detected_pixels = cv2.countNonZero(white_mask)
    #
    # print(f"Gold pixels detected: {detected_pixels}")

    # if detected_pixels > 220:
    #     return True
    # return False


def press_key(key_code, hold_time=0.1):
    win32api.keybd_event(key_code, 0, 0, 0)
    time.sleep(hold_time)
    win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)

def hold_key(key_code, hold_time):
    win32api.keybd_event(key_code, 0, 0, 0)
    time.sleep(hold_time)
    win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)

def restart_game():
    global prev_frame, finisher, holding_direction, counter, finisher_counter, minigame_starter, minigame_starter_counter, vender_counter,not_allowed,pressing
    prev_frame = None
    finisher = False
    minigame_starter = False
    not_allowed = False
    pressing = False
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

def press_for_3_seconds():
    global counter,pressing
    if pressing:
        return
    pressing = True
    press_left_mouse_button()
    time.sleep(3)
    realese_left_moust_button()
    counter = 0
    pressing = False


TEMPLATE_PATHS = [
    "templates/templ1(FINISHER).png",
    "templates/templ2(FINISHER).png",
    "templates/templ3(FINISHER).png",
]

template_names = [path.split('/')[-1] for path in TEMPLATE_PATHS]
templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in TEMPLATE_PATHS]

template2 = cv2.imread("templates/anotherTest.png", cv2.IMREAD_GRAYSCALE)
template3 = cv2.imread("templates/TEST2.png", cv2.IMREAD_GRAYSCALE)


cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FPS, 40)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


model = YOLO("runs/detect/train/weights/best.pt")

CENTER_TOLERANCE = 0.10
prev_frame = None
finisher = False
pressing = False
holding_direction = ""

minigame_starter = False
fishing_active = False
not_allowed = False
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



    if fishing_active:
        if first_time:
            restart_game()
            first_time = False

        position, new_frame = find_bobber(frame_copy)

        if position != None and minigame_starter == False:
            minigame_starter_counter += 1
            if minigame_starter_counter >= 10:
                minigame_starter = True

        if minigame_starter:
            if not pressing:
                minigame(frame_copy, position)

            finisher = is_fishing_finished(frame, template2, threshold=0.5)

            if position == None:
                counter += 1
            else:
                counter = 0
                realese_left_moust_button()

            if finisher:
                finisher_counter += 1
                not_allowed = True
            else:
                finisher_counter = 0
                not_allowed = False

            if counter >= 5 and not not_allowed:
                pressing = True
                press_left_mouse_button()
            else:
                pressing = False
                realese_left_moust_button()

            if finisher_counter >= 2 or counter > 5000:
                print(counter)
                realese_left_moust_button()
                press_righ_mouse_button()
                if vender_counter >= 5:
                    vender_counter = 0
                    replay_thread = recorder.on_execute(Key.f3)

                    if replay_thread:
                        replay_thread.join()

                time.sleep(3)
                restart_game()
                vender_counter += 1


    cv2.imshow('Fishing Mini-Game', new_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()