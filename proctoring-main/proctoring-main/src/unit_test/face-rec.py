import cv2
import face_recognition
import numpy as np

# Load known face encodings and their names
known_face_encodings = []
known_face_names = []

# Load images of known faces
def load_known_faces():
    # Example: Load images from a directory
    # You can replace these with your own images
    image_files = ["person1.jpg", "person2.jpg"]  # Add your image file names here
    names = ["Person 1", "Person 2"]  # Corresponding names

    for image_file, name in zip(image_files, names):
        image = face_recognition.load_image_file(image_file)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Load known faces
load_known_faces()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face to known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()