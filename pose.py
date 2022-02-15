import cv2
import mediapipe as mp
import numpy as np
import time



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



class poseDetector:
    

    def __init__(self,file):
        self.cap = cv2.VideoCapture(file)
        self.time = time.time()
        self.start=False
        
    def set_time(self):
        self.time = time.time()
        self.start=True
    
    
    def get_time(self):
        return int(time.time()-self.time)
    
    
    def get_angle(self):
        
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                 
                    break
                
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
               
               
               
                LEFT_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x
                LEFT_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
                
                
                LEFT_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
                LEFT_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
                
                    
                LEFT_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
                LEFT_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
               
                
                hip=np.array([LEFT_hip_x,LEFT_hip_y])
                knee=np.array([LEFT_knee_x,LEFT_knee_y])
                ankle=np.array([LEFT_ankle_x,LEFT_ankle_y]) 
                
                knee_to_hip = hip-knee
                knee_to_ankle = ankle-knee
                
                dot_between = np.dot(knee_to_hip,knee_to_ankle)/ (np.linalg.norm(knee_to_hip) * np.linalg.norm(knee_to_ankle))
                angle = int(np.degrees(np.arccos(dot_between)))
                print("angle between ==>",angle,self.start,self.get_time())      
                
                #print(LEFT_ankle_x, LEFT_ankle_y)

                
                 
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                
            
                if angle<130 and self.start==False:
                       
                        self.set_time()
                else:
                    
                    if  angle>130 and self.get_time()< 8:
                     
                        cv2.putText(image, 'keep your Knee bent', (125,50), cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, (255,0,0), 2, cv2.LINE_AA)
                        self.start=True
                        self.time=time.time()
                   
                    elif self.get_time()>8:
                        
                        cv2.putText(image, 'Exercise completed!', (125,50), cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, (255,0,0), 2, cv2.LINE_AA)
                        print('exercise completed!, starting again')
                       
                        self.start=True
                        self.time=time.time()
                        cv2.waitKey(2000)
                    
                
                cv2.flip(image,0)
                cv2.imshow('Pose',image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            self.cap.release()


if __name__ == '__main__':
    data = poseDetector('KneeBend.mp4')
    data.get_angle()