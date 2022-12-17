import cv2  # image processing
import mediapipe as mp  # framework to get our pose estimation
from save_landmarks_to_csv import save_landmarks
import pickle
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

count = 0
direction = "going up" # can also be "going down"
word_pred = "none"
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Gray Box
        cv2.rectangle(image, (0,0), (200, 200),
                      (100, 100, 100), cv2.FILLED)

        results = pose.process(image)
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                           mp_pose.POSE_CONNECTIONS)

        h, w, c = image.shape

        if results.pose_landmarks:
            landmark_list = []
            for id, landmark, in enumerate(results.pose_landmarks.landmark):
                landmark_list.append([int(landmark.x * w), int(landmark.y * h)])
            
            landmark_list = np.array(landmark_list).flatten().reshape(1,-1).tolist() # .reshape(1,-1) converts a 1d array into a 2d array (given by error recommendation)

            # print("landmark list", landmark_list)
            if landmark_list:
                pred = model.predict(landmark_list)
                print("pred", pred)

                word_pred = "down" if pred == 0 else "up"

                # Count Reps
                if word_pred == "up":
                    # going up and reached up state
                    if direction == "going up":
                        count += 0.5
                        direction = "going down"
                if word_pred == "down":
                    color = (0, 255, 0)
                    # going down and reached down state
                    if direction == "going down":
                        count += 0.5
                        direction = "going up"

        else:
            word_pred = "none"
                
        # Up or Down Status
        cv2.putText(image, word_pred, (35, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)

        # Reps
        cv2.putText(image, f"Reps: {int(count)}", (35, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Webcam", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()