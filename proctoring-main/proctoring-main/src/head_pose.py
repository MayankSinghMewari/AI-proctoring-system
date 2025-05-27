from audioop import avg
from glob import glob
from itertools import count
import cv2
import mediapipe as mp
import numpy as np
import threading as th
import sounddevice as sd
import audio
import os



# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Define thresholds for eye movement
EYE_MOVEMENT_THRESHOLD = 0.2  # Adjust this value based on your requirements

def detect_eye_movement(eye_landmarks):
    # Calculate the center of the eyes
    left_eye_center = np.mean(eye_landmarks[0:6], axis=0)  # Average of left eye landmarks
    right_eye_center = np.mean(eye_landmarks[6:12], axis=0)  # Average of right eye landmarks

    # Calculate the distance from the center of the face (you can define a fixed point or use landmarks)
    face_center = np.array([0.5, 0.5])  # Assuming normalized coordinates (0 to 1)

    left_eye_distance = np.linalg.norm(left_eye_center - face_center)
    right_eye_distance = np.linalg.norm(right_eye_center - face_center)

    # Check if the distance exceeds the threshold
    if left_eye_distance > EYE_MOVEMENT_THRESHOLD or right_eye_distance > EYE_MOVEMENT_THRESHOLD:
        return True  # Eye movement detected
    return False  # No suspicious eye movement


def pose():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks and perform detection
                eye_landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark[33:45]])  # Left eye

                # Draw eye landmarks (dots)
                for lm in eye_landmarks:
                    x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Draw eye dots in red

                # Assuming you have nose landmarks defined
                nose_2d = (face_landmarks.landmark[1].x * image.shape[1], face_landmarks.landmark[1].y * image.shape[0])
                nose_3d_projection = ...  # Calculate or define this based on your logic

                # Draw the nose line
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                cv2.line(image, p1, p2, (0, 255, 0), 2)  # Draw nose line in green

        else:
            print("No faces detected.")

        cv2.imshow('Head Pose Estimation', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()




# Load the pre-trained YOLO model
net = cv2.dnn.readNet("D:/resetart 2/proctoring-main/proctoring-main/src/yolov3.weights", "D:/resetart 2/proctoring-main/proctoring-main/src/yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()

# Check if the output is a scalar or an array
if isinstance(unconnected_out_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
else:
    output_layers = [layer_names[unconnected_out_layers - 1]]


# Load class names
with open("D:/resetart 2/proctoring-main/proctoring-main/src/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_smartphone(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:  # Confidence threshold
                if classes[class_id] == "cell phone":  # Check if the detected class is a smartphone
                    # Draw bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    color = (0,0,255)
                    cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), 
                                  (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                    return True  # Smartphone detected
    return False  # No smartphone detected

def pose():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Detect smartphone
        if detect_smartphone(image):
            print("Cheating detected: Smartphone is visible.")
        

# place holders and global variables
x = 0                                       # X axis head pose
y = 0                                       # Y axis head pose

X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0

def pose():
    global VOLUME_NORM, x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions

    while cap.isOpened():
        success, image = cap.read()
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False
        
        # Get the result
        results = face_mesh.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        
        face_ids = [33, 263, 1, 61, 291, 199]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)
                
                for idx, lm in enumerate(face_landmarks.landmark):
                    # print(lm)
                    if idx in face_ids:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # print(y)

                # See where the user's head tilting
                if y < -8:
                    text = "Looking Left"
                elif y > 8:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"
                text = str(int(x)) + "::" + str(int(y)) + text
                # print(str(int(x)) + "::" + str(int(y)))
                # print("x: {x}   |   y: {y}  |   sound amplitude: {amp}".format(x=int(x), y=int(y), amp=audio.SOUND_AMPLITUDE))
                
                # Y is left / right
                # X is up / down
                if y < -8 or y > 8:
                    X_AXIS_CHEAT = 1
                else:
                    X_AXIS_CHEAT = 0

                if x < -5 or x > 5:
                    Y_AXIS_CHEAT = 1
                else:
                    Y_AXIS_CHEAT = 0

                # print(X_AXIS_CHEAT, Y_AXIS_CHEAT)
                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                   

                # Add the text on the image
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

if __name__ == "__main__":
    t1 = th.Thread(target=pose)

    t1.start()

    t1.join()