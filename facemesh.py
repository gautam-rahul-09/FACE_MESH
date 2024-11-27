import mediapipe as mp
import numpy as np
import cv2

# Initialize webcam and MediaPipe Face Mesh
cap = cv2.VideoCapture(0)

facmesh = mp.solutions.face_mesh
face = facmesh.FaceMesh(static_image_mode = True, min_tracking_confidence= 0.6, min_detection_confidence = 0.6 )
draw = mp.solutions.drawing_utils


while True:
    _, frm = cap.read()
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
 
   # Process the frame with Face Mesh   
    op = face.process(rgb)
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            draw.draw_landmarks(image=frm,
                landmark_list=i,
                connections=facmesh.FACEMESH_TESSELATION,  # Update based on available connections
                landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=1))
    
    cv2.imshow("Window",frm)
    
    if cv2.waitKey(1) == 27:   # ESC key to break the loop
        cap.release()
        cv2.destroyAllWindows()
        break


