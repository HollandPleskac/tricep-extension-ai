import cv2
import mediapipe as mp
import time
from save_landmarks_to_csv import save_landmarks

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(image)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('MediaPipe Pose', image)

        k = cv2.waitKey(1)
        if k & 0xFF == ord('u'):
          print("saving landmarks up")
          save_landmarks(image, "up", results.pose_landmarks)
        if k & 0xFF == ord('d'):
          print("saving landmarks down")
          save_landmarks(image, "down", results.pose_landmarks)

        if k & 0xFF == ord('q'):
            break
        

cap.release()
cv2.destroyAllWindows()