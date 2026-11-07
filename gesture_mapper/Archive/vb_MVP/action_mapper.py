# action_mapper.py
# Maps gestures to system input actions (e.g. keyboard/mouse)

import time
import yaml
import pyautogui

CONFIG_PATH = "gesture_config.yaml"

def load_mappings():
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

def perform_action(action):
    print(f"[Mapper] perform_action called with: {action}") # DEBUG PRINT
    if isinstance(action, str):
        if action.startswith("key:"):
            key = action.split(":", 1)[1]
            print(f"[Mapper] Attempting to press key: {key}") # DEBUG PRINT
            pyautogui.press(key)
        elif action.startswith("hold:"):
            key = action.split(":", 1)[1]
            print(f"[Mapper] Attempting to hold key: {key}") # DEBUG PRINT
            pyautogui.keyDown(key)
        elif action.startswith("release:"):
            key = action.split(":", 1)[1]
            print(f"[Mapper] Attempting to release key: {key}") # DEBUG PRINT
            pyautogui.keyUp(key)
        elif action.startswith("click:"):
            button = action.split(":", 1)[1]
            print(f"[Mapper] Attempting to click button: {button}") # DEBUG PRINT
            pyautogui.click(button=button)
    else:
        print(f"[Mapper] Action '{action}' is not a string, cannot perform.") # DEBUG PRINT

def map_and_perform(gesture_name, mappings):
    print(f"[Mapper] map_and_perform called with: {gesture_name}, Mappings: {mappings}") # DEBUG PRINT
    if gesture_name in mappings:
        action = mappings[gesture_name]
        print(f"[Mapper] Found action '{action}' for gesture '{gesture_name}'") # DEBUG PRINT
        perform_action(action)
    else:
        print(f"[Mapper] Gesture '{gesture_name}' has no mapped action.") # DEBUG PRINT

if __name__ == "__main__":
    mappings = load_mappings()
    test_gesture = "raise_hand"
    map_and_perform(test_gesture, mappings)
