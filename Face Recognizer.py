#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Generate Dataset
# 2. Train the classifier and save it
# 3. Detect the face and name it if it is already stored in our dataset


# # Generate Dataset
# 

# In[2]:


import cv2


# In[3]:


def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor = 1.3
        # minimum neighbor = 5
         
        if faces == ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
     
    cap = cv2.VideoCapture(0)
    id =1
    img_id = 0
     
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
             
            cv2.imshow("Cropped face", face)
             
        if cv2.waitKey(1)==13 or int(img_id)==200: #13 is the ASCII character of Enter
            break
             
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed....")
# generate_dataset()


# # Train the classifier and save it
# 

# In[6]:


import numpy as np
from PIL import Image
import os
import cv2


# In[8]:


def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
     
    faces = []
    ids = []
     
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
         
        faces.append(imageNp)
        ids.append(id)
         
    ids = np.array(ids)
     
    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
train_classifier("data")


# # Detect the face and name it if it is already in the dataset
# 

# In[9]:


import cv2
import numpy as np
from PIL import Image
import os


# In[ ]:


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

    coords = []
    
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
         
        id, pred = clf.predict(gray_image[y:y+h,x:x+w])
        confidence = int(100*(1-pred/300))
         
        if confidence>75:
            if id == 1:
                cv2.putText(img, "Anurag", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            if id == 2:
                cv2.putText(img, "Tom", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

        coords = [x, y, w, h]
    return coords

def recognize(img, clf, faceCascade):
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
    return img


# loading classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")
 
video_capture = cv2.VideoCapture(0)
 
while True:
    ret, img = video_capture.read()
    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face Detection", img)
     
    if cv2.waitKey(1) == 13:
        break
        
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




