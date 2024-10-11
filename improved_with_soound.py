import os
import cv2
import numpy as np
import time
import pygame
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('mrl_eye_state_model.h5')

# Parameters
img_size = 64  # Size to which the images will be resized
closed_eye_time = None  # To track when eyes were first detected as closed
alert_played_1 = False  # To prevent repeatedly playing the first sound
alert_played_2 = False  # To prevent repeatedly playing the second sound

# Load the pre-trained Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize pygame for sound playback
pygame.mixer.init()

# Real-time detection using the webcam
def detect_eye_state():
    global closed_eye_time, alert_played_1, alert_played_2

    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale for eye detection
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        predictions = []
        overall_label = "Unknown"  # Initialize with a default value

        # Loop through detected eyes
        for (x, y, w, h) in eyes[:2]:  # Only look at the first two detected eyes
            # Crop the eye region from the frame
            eye_img = frame[y:y+h, x:x+w]

            # Resize and preprocess the eye image for prediction
            eye_img = cv2.resize(eye_img, (img_size, img_size))
            eye_img = eye_img / 255.0
            eye_img = np.expand_dims(eye_img, axis=0)  # Add batch dimension

            # Make prediction using the trained model
            prediction = model.predict(eye_img)
            predictions.append(prediction[0][0])  # Store the prediction

        # Calculate the mean prediction for both eyes
        if len(predictions) > 0:
            mean_prediction = np.mean(predictions)

            # Classify as "Open" or "Closed"
            if mean_prediction < 0.5:
                overall_label = "Closed"

                # Start tracking time when eyes first close
                if closed_eye_time is None:
                    closed_eye_time = time.time()  # Record the time when eyes first closed

                # Check how long eyes have been closed
                elapsed_time = time.time() - closed_eye_time

                # Play the first sound at 1.5 seconds of closed eyes
                if elapsed_time > 1.5 and not alert_played_1:
                    print("Playing first sound: beep.wav")  # Debug print
                    pygame.mixer.music.load(r'D:\Projects\DD-Korean tutor from YT\beep.wav')
                    pygame.mixer.music.play()
                    alert_played_1 = True  # Mark first sound as played

                # Play the second sound at 2.5 seconds in an infinite loop
                if elapsed_time > 2.5 and not alert_played_2:
                    print("Playing second sound: horn.  in loop")  # Debug print
                    pygame.mixer.music.load(r'D:\Projects\DD-Korean tutor from YT\horn.mp3')
                    pygame.mixer.music.play(loops=-1)  # Infinite loop
                    alert_played_2 = True  # Mark second sound as played

            else:
                overall_label = "Open"
                closed_eye_time = None  # Reset when eyes are open
                alert_played_1 = False  # Reset alert states when eyes open
                alert_played_2 = False
                pygame.mixer.music.stop()  # Stop sound if eyes open

        # Display the overall result
        cv2.putText(frame, f'Eyes: {overall_label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the video feed with detection
        cv2.imshow('Eye State Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time detection
detect_eye_state()
