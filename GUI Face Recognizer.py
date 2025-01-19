#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import pymysql


# In[ ]:


window = tk.Tk()
window.title("Face Recognition System")
window.geometry("900x600")
window.configure(bg="#ffffff")

# Header Section
header_frame = tk.Frame(window, bg="#4A90E2", height=100)
header_frame.pack(fill=tk.X)

header_label = tk.Label(header_frame, text="Face Recognition System", font=("Arial", 30, "bold"), bg="#4A90E2", fg="white")
header_label.pack(pady=20)

# Add an icon image for the header (you can place an image of your choice in the current directory)
try:
    icon_image = Image.open("icon.png")  # Replace with your icon file path
    icon_image = icon_image.resize((60, 60), Image.ANTIALIAS)
    icon_photo = ImageTk.PhotoImage(icon_image)
    icon_label = tk.Label(header_frame, image=icon_photo, bg="#4A90E2")
    icon_label.place(x=20, y=20)
except:
    pass  # icon image will be skipped if not found

# Form Section
form_frame = tk.Frame(window, bg="#ffffff")
form_frame.pack(pady=30)

# Name label and entry
name_label = tk.Label(form_frame, text="Name:", font=("Arial", 16), bg="#ffffff", fg="#333333")
name_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
name_entry = tk.Entry(form_frame, width=30, font=("Arial", 14), bd=3)
name_entry.grid(row=0, column=1, padx=10, pady=10)

# Age label and entry
age_label = tk.Label(form_frame, text="Age:", font=("Arial", 16), bg="#ffffff", fg="#333333")
age_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
age_entry = tk.Entry(form_frame, width=30, font=("Arial", 14), bd=3)
age_entry.grid(row=1, column=1, padx=10, pady=10)

# Address label and entry
address_label = tk.Label(form_frame, text="Address:", font=("Arial", 16), bg="#ffffff", fg="#333333")
address_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
address_entry = tk.Entry(form_frame, width=30, font=("Arial", 14), bd=3)
address_entry.grid(row=2, column=1, padx=10, pady=10)

# Button Section
button_frame = tk.Frame(window, bg="#ffffff")
button_frame.pack(pady=20)

# Function placeholders (to be connected to actual functionalities later)
def generate_dataset():
    if name_entry.get() == "" or age_entry.get() == "" or address_entry.get() == "":
        messagebox.showinfo("Result", "Please provide complete details of the user")
    else:
        # Connect to the database
        mydb = pymysql.connect(host="localhost", user="root", password="dhuri", database="authorized_user")
        mycursor = mydb.cursor()

        # Insert the new user data into the table
        sql = "INSERT INTO authorized_user_table (name, age, address) VALUES (%s, %s, %s)"
        val = (name_entry.get(), age_entry.get(), address_entry.get())
        mycursor.execute(sql, val)
        mydb.commit()

        # Retrieve the unique id of the inserted user
        mycursor.execute("SELECT LAST_INSERT_ID()")
        user_id = mycursor.fetchone()[0]  # Get the newly assigned user ID
        # print("User ID:", user_id)  # Debugging step to verify the user ID

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:  # Corrected the 'faces is ()' syntax warning
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y+h, x:x+w]
            return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Use the unique user ID in the filename
                file_name_path = f"data/user.{user_id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)

                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped face", face)

            # Break when Enter key is pressed or 200 images have been captured
            if cv2.waitKey(1) == 13 or img_id == 200:  # 13 is ASCII for Enter
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Result", "Generating Dataset Completed")


def train_classifier():
    data_dir = r".\data"
    
    # Ensure the directory exists
    if not os.path.exists(data_dir):
        messagebox.showerror("Error", f"Data directory not found: {data_dir}")
        return
    
    # Initialize lists for faces and IDs
    faces = []
    ids = []
    
    try:
        # Iterate over files in the directory
        for image in os.listdir(data_dir):
            image_path = os.path.join(data_dir, image)
            
            # Validate file format (ensure it's an image file)
            if not image.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Load and process the image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            imageNp = np.array(img, 'uint8')  # Convert to NumPy array
            
            # Extract the ID from the filename
            try:
                id = int(os.path.split(image_path)[1].split(".")[1])
            except (IndexError, ValueError):
                print(f"Skipping invalid file: {image_path}")
                continue
            
            faces.append(imageNp)
            ids.append(id)
        
        # Ensure there are faces to train on
        if len(faces) == 0 or len(ids) == 0:
            messagebox.showerror("Error", "No valid training data found.")
            return
        
        ids = np.array(ids)
        
        # Train and save the classifier
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")
        messagebox.showinfo("Result", "Training Dataset Completed")
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")

    

def detect_faces():
    # Function to draw a boundary around the detected face
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
    
        coords = []
        
        for (x, y, w, h) in features:
            # Predict the identity
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
            
            mydb = pymysql.connect(host="localhost", user="root", password="dhuri", database="authorized_user")
            mycursor = mydb.cursor()
            mycursor.execute("select name from authorized_user_table where id=" + str(id))
            result = mycursor.fetchone()
            
            # Check if the name exists in the database
            if result:
                name = ''.join(result)  # Convert the tuple result to a string
            else:
                name = "UNKNOWN"  # Fallback if no name is found
            
            # Set the name and confidence text based on the ID and confidence level
            if confidence > 75:
                text = f"{name} {confidence}%"
                color = (0, 255, 0)  # Green for identified persons
            else:
                text = f"UNKNOWN {confidence}%"
                color = (0, 0, 255)  # Red for unknown faces
            
            # Draw rectangle and text around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            
            # Store the coordinates of the rectangle
            coords = [x, y, w, h]
        
        return coords
        
    # Function to recognize the face and draw a bounding box
    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), clf)
        return img
        
    # Load the face classifier and trained model
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")  # Load the trained face recognizer model
    
    # Capture video from the webcam
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        
        # Display the processed image in a window
        cv2.imshow("Face Detection", img)
        
        # Break the loop if Enter (key code 13) is pressed
        if cv2.waitKey(1) == 13:
            break        
    
    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()


# Buttons
generate_btn = tk.Button(button_frame, text="Generate Dataset", font=("Arial", 14, "bold"), bg="#FFA500", fg="white",
                         width=18, height=2, command=generate_dataset)
generate_btn.grid(row=0, column=0, padx=15, pady=10)

train_btn = tk.Button(button_frame, text="Train Classifier", font=("Arial", 14, "bold"), bg="#4CAF50", fg="white",
                      width=18, height=2, command=train_classifier)
train_btn.grid(row=0, column=1, padx=15, pady=10)

detect_btn = tk.Button(button_frame, text="Detect Faces", font=("Arial", 14, "bold"), bg="#FF69B4", fg="white",
                       width=18, height=2, command=detect_faces)
detect_btn.grid(row=0, column=2, padx=15, pady=10)

# Display Window Instructions
info_label = tk.Label(window, text="Instructions:\n1. Enter your details.\n2. Click 'Generate Dataset' to capture face data.\n3. Click 'Train Classifier' to train the model.\n4. Click 'Detect Faces' to start detection.", 
                      font=("Arial", 12), bg="#ffffff", fg="#333333", justify="left")
info_label.pack(pady=20)

window.mainloop()
