import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from playsound import playsound  # Import playsound to play an alert tone
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from playsound import playsound

# Load the trained YOLO model
@st.cache_resource
def load_yolo_model():
    model = YOLO('best.pt')  # Adjust the model path accordingly
    return model

# Initialize YOLO model
model = load_yolo_model()

def predict_image(image):
    results = model.predict(image, conf=0.25, iou=0.45, save=False, show=False)
    return results

def display_results_as_dataframe(results):
    boxes = results[0].boxes
    df = pd.DataFrame(boxes.xywh.cpu().numpy(), columns=['x_center', 'y_center', 'width', 'height'])
    df['confidence'] = boxes.conf.cpu().numpy()
    df['class_id'] = boxes.cls.cpu().numpy()
    df['class_name'] = [model.names[int(cls)] for cls in df['class_id']]  # Get class names from model
    return df

# def send_email(image_path):
#     sender_email = "your_email@gmail.com"
#     receiver_email = "recipient_email@gmail.com"
#     password = "your_password"
    
#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = receiver_email
#     msg['Subject'] = "Weapon Detected Alert"
#     body = "A weapon has been detected. Please find the attached image."
#     msg.attach(MIMEText(body, 'plain'))
    
#     attachment = open(image_path, "rb")
#     part = MIMEBase('application', 'octet-stream')
#     part.set_payload((attachment).read())
#     encoders.encode_base64(part)
#     part.add_header('Content-Disposition', "attachment; filename= " + os.path.basename(image_path))
#     msg.attach(part)
    
#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.starttls()
#     server.login(sender_email, password)
#     text = msg.as_string()
#     server.sendmail(sender_email, receiver_email, text)
#     server.quit()
    
#     st.write("Email sent successfully!")

from pydub import AudioSegment
from pydub.playback import play
import os
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play

def play_alert_sound(file_path):
    sound = AudioSegment.from_file(file_path, format="wav")
    play(sound)

def check_for_weapons(results, image):
    weapon_classes = ["knife", "gun", "rifle", "Weapon"]
    detected = False  # Flag to track if any weapon is detected

    for result in results[0].boxes.cls:
        class_name = model.names[int(result)]
        if class_name in weapon_classes:
            detected = True  # Set the flag to True

            # Generate a unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detected_weapon_path = f"detected_weapon_{timestamp}.jpg"

            # Ensure the directory exists
            os.makedirs("detected_images", exist_ok=True)
            full_path = os.path.join("detected_images", detected_weapon_path)

            # Save the detected image with a unique filename
            image.save(full_path)
            st.write(f"Weapon detected! Image saved as {full_path}")

            # Play the alert sound
            file_path = os.path.abspath("alert1.wav")
            play_alert_sound(file_path)

    return detected



def run():
    st.title("weapon detection using Computer vision and deep learning")
    st.write("Upload an image or video to predict objects or use your webcam to detect in real-time.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        image_array = np.array(image)
        results = predict_image(image_array)
        
        st.write("Prediction Results:")
        prediction_df = display_results_as_dataframe(results)
        st.write(prediction_df)
        
        if check_for_weapons(results, image):
            st.write("Weapon detected! Alert triggered and email sent.")
        
        img_with_predictions = results[0].plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_with_predictions, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        st.pyplot(plt)

    if 'stop_camera' not in st.session_state:
        st.session_state.stop_camera = False

    if st.button('Start Camera'):
        st.session_state.stop_camera = False

    if st.button('Stop Camera'):
        st.session_state.stop_camera = True

    if not st.session_state.stop_camera:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = predict_image(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if check_for_weapons(results, Image.fromarray(frame)):
                st.write("Weapon detected! Alert triggered and email sent.")
            
            img_with_predictions = results[0].plot()
            stframe.image(img_with_predictions, channels="BGR", use_container_width=True)
            
            if st.session_state.stop_camera:
                cap.release()
                break
        cap.release()

if __name__ == '__main__':
    run()
