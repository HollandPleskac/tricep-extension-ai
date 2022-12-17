import pandas as pd
import numpy as np

from get_cols import get_cols

# Custom object to easily represent landmark information I need
class Landmark:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

def save_landmarks(img, pose_type, pose_landmarks):

  h, w, c = img.shape

  landmark_obj_list = []
  for id, landmark, in enumerate(pose_landmarks.landmark):
    landmark_obj = Landmark(id, int(landmark.x * w), int(landmark.y * h))
    # print("landmark", landmark_obj.id, landmark_obj.x, landmark_obj.y)
    landmark_obj_list.append(landmark_obj)
    
  landmarks= [[obj.x,obj.y] for obj in landmark_obj_list]
  landmarks = np.array(landmarks)
  landmarks = landmarks.flatten() # [ x1, y1, ... , x33, y33] 66 elements total in this list
  # print('flattened', landmarks, len(landmarks))

  cols = get_cols(len(landmark_obj_list))

  df = pd.DataFrame(landmarks.reshape(1, 66), index=[pose_type], columns=cols)
  print(df)
  df.to_csv('landmarks.csv', mode='a', index=[pose_type], header=None) # adds index to csv file