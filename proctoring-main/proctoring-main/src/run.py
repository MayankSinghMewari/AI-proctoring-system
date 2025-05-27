import audio
import head_pose
import detection
import threading
import matplotlib.pyplot as plt
import numpy as np
import time
from ui import ui

PER_SEC_UPDATE = 1
GLOBAL_CHEAT = 0


def dynamic_per_sec_update():
    global PER_SEC_UPDATE
    while True:
        # Logic to adjust PER_SEC_UPDATE dynamically
        if GLOBAL_CHEAT == 1:  # Replace with your actual condition
            PER_SEC_UPDATE = 2  # Update twice per second
        else:
            PER_SEC_UPDATE = 1  # Default to once per second

        time.sleep(1) 


if __name__ == "__main__":
    # Start the UI in a separate thread
    ui_thread = threading.Thread(target=ui.run)
    ui_thread.start()

    # Start the dynamic update thread
    update_thread = threading.Thread(target=dynamic_per_sec_update)
    update_thread.start()

    # Start other threads for head pose and audio detection
    head_pose_thread = threading.Thread(target=head_pose.pose)
    audio_thread = threading.Thread(target=audio.sound)
    detection_thread = threading.Thread(target=detection.run_detection)

    head_pose_thread.start()
    audio_thread.start()
    detection_thread.start()

    head_pose_thread.join()
    audio_thread.join()
    detection_thread.join()
