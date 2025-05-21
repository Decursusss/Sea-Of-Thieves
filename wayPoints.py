import time
import json
import threading
from pynput import keyboard, mouse
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController


class ActionRecorder:
    def __init__(self):
        self.keyboard_controller = KeyboardController()
        self.mouse_controller = MouseController()
        self.recording = False
        self.replaying = False
        self.actions = []
        self.start_time = 0

        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.mouse_listener = mouse.Listener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click,
            on_scroll=self.on_mouse_scroll
        )

        self.keyboard_listener.start()
        self.mouse_listener.start()

    def on_key_press(self, key):
        if key == Key.f1 and not self.recording and not self.replaying:
            print("Запись началась - нажмите F2 для остановки")
            self.start_recording()
            return

        if key == Key.f2 and self.recording:
            print("Запись остановлена")
            self.stop_recording()
            return

        if key == Key.f3 and not self.recording and not self.replaying:
            print("Воспроизведение...")
            threading.Thread(target=self.replay).start()
            return

        if self.recording:
            try:
                key_char = key.char
            except AttributeError:
                key_char = str(key)

            current_time = time.time() - self.start_time
            self.actions.append({
                'type': 'key_press',
                'key': key_char,
                'time': current_time
            })

    def on_execute(self, key):
        if key == Key.f3 and not self.recording and not self.replaying:
            print("Воспроизведение...")
            t = threading.Thread(target=self.replay)
            t.start()
            return t
        return None

    def on_key_release(self, key):
        if self.recording:
            try:
                key_char = key.char
            except AttributeError:
                key_char = str(key)

            current_time = time.time() - self.start_time
            self.actions.append({
                'type': 'key_release',
                'key': key_char,
                'time': current_time
            })

    def on_mouse_move(self, x, y):
        if self.recording:
            current_time = time.time() - self.start_time
            if not self.actions or current_time - self.actions[-1].get('time', 0) > 0.05:
                last_x = self.actions[-1].get('x', x) if self.actions else x
                last_y = self.actions[-1].get('y', y) if self.actions else y
                rel_x = x - last_x
                rel_y = y - last_y

                self.actions.append({
                    'type': 'mouse_move',
                    'x': x,
                    'y': y,
                    'rel_x': rel_x,
                    'rel_y': rel_y,
                    'time': current_time
                })

    def on_mouse_click(self, x, y, button, pressed):
        if self.recording:
            current_time = time.time() - self.start_time
            btn = 'left' if button == Button.left else 'right' if button == Button.right else 'middle'
            action_type = 'mouse_press' if pressed else 'mouse_release'

            self.actions.append({
                'type': action_type,
                'button': btn,
                'x': x,
                'y': y,
                'time': current_time
            })

    def on_mouse_scroll(self, x, y, dx, dy):
        if self.recording:
            current_time = time.time() - self.start_time
            self.actions.append({
                'type': 'mouse_scroll',
                'x': x,
                'y': y,
                'dx': dx,
                'dy': dy,
                'time': current_time
            })

    def start_recording(self):
        self.recording = True
        self.actions = []
        self.start_time = time.time()

    def stop_recording(self):
        self.recording = False

    def replay(self):
        if not self.actions:
            print("Нет действий для воспроизведения")
            return

        self.replaying = True
        previous_time = 0

        for action in self.actions:
            time_to_wait = action['time'] - previous_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            previous_time = action['time']

            if action['type'] == 'key_press':
                key = self.parse_key(action['key'])
                self.keyboard_controller.press(key)

            elif action['type'] == 'key_release':
                key = self.parse_key(action['key'])
                self.keyboard_controller.release(key)

            elif action['type'] == 'mouse_move':
                if 'rel_x' in action and 'rel_y' in action:
                    self.mouse_controller.move(action['rel_x'], action['rel_y'])
                else:
                    self.mouse_controller.position = (action['x'], action['y'])

            elif action['type'] == 'mouse_press':
                self.mouse_controller.position = (action['x'], action['y'])
                button = Button.left if action['button'] == 'left' else Button.right if action[
                                                                                            'button'] == 'right' else Button.middle
                self.mouse_controller.press(button)

            elif action['type'] == 'mouse_release':
                self.mouse_controller.position = (action['x'], action['y'])
                button = Button.left if action['button'] == 'left' else Button.right if action[
                                                                                            'button'] == 'right' else Button.middle
                self.mouse_controller.release(button)

            elif action['type'] == 'mouse_scroll':
                self.mouse_controller.scroll(action['dx'], action['dy'])

        print("Воспроизведение завершено")
        self.replaying = False

    def parse_key(self, key_str):
        special_keys = {
            'Key.shift': Key.shift,
            'Key.alt': Key.alt,
            'Key.ctrl': Key.ctrl,
            'Key.cmd': Key.cmd,
            'Key.space': Key.space,
            'Key.tab': Key.tab,
            'Key.enter': Key.enter,
            'Key.backspace': Key.backspace,
            'Key.esc': Key.esc,
            'Key.delete': Key.delete,
            'Key.up': Key.up,
            'Key.down': Key.down,
            'Key.left': Key.left,
            'Key.right': Key.right,
        }

        if key_str in special_keys:
            return special_keys[key_str]
        elif key_str.startswith('Key.'):
            key_name = key_str.split('.')[1]
            try:
                return getattr(Key, key_name)
            except AttributeError:
                print(f"Неизвестная клавиша: {key_str}")
                return key_str
        else:
            return key_str

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.actions, f)
        print(f"Запись сохранена в {filename}")

    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                self.actions = json.load(f)
            print(f"Загружено {len(self.actions)} действий из {filename}")
        except FileNotFoundError:
            print(f"Файл {filename} не найден")
        except json.JSONDecodeError:
            print(f"Ошибка при чтении файла {filename}")