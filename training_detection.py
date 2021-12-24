import cv2
import mediapipe as mp
import numpy as np



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#Set output media

cap= cv2.VideoCapture(0)
#cap = cv2.VideoCapture("Dataset/Dumbell_hammer_curls/video/WhatsApp Video 2021-12-24 at 11.29.01.mp4")
#cap = cv2.VideoCapture("Dataset/Dumbell_biceps_curls/video/VID20211209193829.mp4")
#cap = cv2.VideoCapture("Dataset/Push_up/video/WhatsApp Video 2021-12-24 at 11.15.08.mp4")

# Get the Default resolutions for ouput video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the code and filename.
out = cv2.VideoWriter('bicep_curls_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Curl counter variables
counter = 0
stage = None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)


            # Curl counter logic
            if angle > 150:
                stage = "down"
            if angle < 20 and stage == 'down':
                stage = "up"
                counter += 1
                print(counter)

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (134, 39, 134), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                                  )

        landmarks = []
        height, width, _ = image.shape
        if results.pose_landmarks:

            # Draw Pose landmarks on the output image.
            mp_drawing.draw_landmarks(image, landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS)

            # Iterate over the detected landmarks.
            for landmark in results.pose_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
            # Initialize the label of the pose. It is not known at this stage.
            label = 'Unknown Pose'

            # Specify the color (Red) with which the label will be written on the image.
            color = (0, 0, 255)

            # Get the angle between the left shoulder, elbow and wrist points.
            left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

            right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

            right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

            # Biceps curls label
            if left_elbow_angle > 0 and left_elbow_angle < 40 and right_elbow_angle > 0 and right_elbow_angle < 40:
                label = 'Bicep Curls'

            if left_elbow_angle > 40 and left_elbow_angle < 70:
                label = 'Right Hammer Curls'

            if right_elbow_angle > 40 and right_elbow_angle < 70:
                label = 'Left Hammer Curls'

            if right_shoulder_angle > 40 and right_shoulder_angle < 70:
                label = 'Push Up'

            # Check if the pose is classified successfully
            if label != 'Unknown Pose':
                # Update the color (to green) with which the label will be written on the image.
                color = (0, 255, 0)

                # Write the label on the output image.
            cv2.putText(image, label, (0, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

        cv2.imshow('Fitness Training Counter and Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()