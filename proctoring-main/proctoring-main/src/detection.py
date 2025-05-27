import time
import audio
import head_pose
import matplotlib.pyplot as plt
import numpy as np
import threading
from ui import ui

PLOT_LENGTH = 200

# place holders 
GLOBAL_CHEAT = 0
PERCENTAGE_CHEAT = 0
CHEAT_THRESH = 0.6
XDATA = list(range(200))
YDATA = [0]*200
PER_SEC_UPDATE = 1 

cheating_reasons = []

def avg(current, previous):
    if previous > 1:
        return 0.65
    if current == 0:
        if previous < 0.01:
            return 0.01
        return previous / 1.01
    if previous == 0:
        return current
    return 1 * previous + 0.1 * current

def process():
    global GLOBAL_CHEAT, PERCENTAGE_CHEAT, CHEAT_THRESH, cheating_reasons #head_pose.X_AXIS_CHEAT, head_pose.Y_AXIS_CHEAT, audio.AUDIO_CHEAT
    # print(head_pose.X_AXIS_CHEAT, head_pose.Y_AXIS_CHEAT)
    # print("entered proess()...")

    if 'general_cheating_message_displayed' not in globals():
        general_cheating_message_displayed = False

    if GLOBAL_CHEAT == 0:
        if head_pose.X_AXIS_CHEAT == 0:
            if head_pose.Y_AXIS_CHEAT == 0:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.2, PERCENTAGE_CHEAT)
            else:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.2, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.4, PERCENTAGE_CHEAT)
        else:
            if head_pose.Y_AXIS_CHEAT == 0:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.1, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.4, PERCENTAGE_CHEAT)
            else:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.15, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.25, PERCENTAGE_CHEAT)
    else:
        if head_pose.X_AXIS_CHEAT == 0:
            if head_pose.Y_AXIS_CHEAT == 0:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.55, PERCENTAGE_CHEAT)
            else:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.55, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)
        else:
            if head_pose.Y_AXIS_CHEAT == 0:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.6, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)
            else:
                if audio.AUDIO_CHEAT == 0:
                    PERCENTAGE_CHEAT = avg(0.5, PERCENTAGE_CHEAT)
                else:
                    PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)
    if head_pose.X_AXIS_CHEAT == 1:
        cheating_reasons.append("Head movement detected.")
    
    if audio.AUDIO_CHEAT == 1:
        cheating_reasons.append("Suspicious audio detected.")
    
    if GLOBAL_CHEAT == 1:
        cheating_reasons.append("General cheating detected.")

    # Check if cheating percentage exceeds threshold
    if PERCENTAGE_CHEAT > CHEAT_THRESH:
        GLOBAL_CHEAT = 1
        
        if not general_cheating_message_displayed:
            notification_message = "Suspicious Activity Detected!\nReasons:\n"
            notification_message += "\n".join(cheating_reasons)
            ui.show_notification(notification_message)
            general_cheating_message_displayed = True  # Set the flag to True after displaying the message
            
        else:
            GLOBAL_CHEAT = 0
            cheating_reasons.clear()  # Clear reasons if no cheating is detected
            general_cheating_message_displayed = False 



def run_detection():
    global XDATA,YDATA
    plt.show()
    axes = plt.gca()
    axes.set_xlim(0, 200)
    axes.set_ylim(0,1)
    line, = axes.plot(XDATA, YDATA, 'r-')
    plt.title("SUSpicious Behaviour Detection")
    plt.xlabel("Time")
    plt.ylabel("Cheat Probablity")
    i = 0
    while True:
        YDATA.pop(0)
        YDATA.append(PERCENTAGE_CHEAT)
        line.set_xdata(XDATA)
        line.set_ydata(YDATA)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(1/5)
        process()

def dynamic_per_sec_update():
    global PER_SEC_UPDATE
    while True:
        # Logic to adjust PER_SEC_UPDATE dynamically
        if GLOBAL_CHEAT == 1:  # Replace with your actual condition
            PER_SEC_UPDATE = 2  # Update twice per second
        else:
            PER_SEC_UPDATE = 1  # Default to once per second

        time.sleep(1)
