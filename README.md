# GUI Face Recognizer System

## Overview

The Face Recognition System is a Python-based desktop application for face detection, recognition, and user management. It uses OpenCV for image processing, MySQL for user data storage, and Tkinter for the graphical user interface (GUI). The system allows users to capture face datasets, train a classifier, and detect faces in real time.

## Features
- *User Registration:*
Enter and save user details (name, age, address) to a MySQL database.
- *Dataset Generation:* 
Capture and store face images using the webcam.
- *Classifier Training:* 
Train an LBPH (Local Binary Patterns Histograms) face recognizer model using the captured dataset.
- *Real-time Face Detection:*
Detect and recognize faces with confidence levels and display user names.

## Prerequisites
- Python 3.10 or later
- MySQL database server
- Required Python libraries:
    - tkinter
    - Pillow
    - opencv-python
    - opencv-contrib-python
    - numpy
    - pymysql

## Installation
- Clone this repository or download the project files.
- Create and Activate a Virtual Environment:  
    - On Windows:  
```python -m venv myenv```  
```myenv\Scripts\activate```
    - On macOS/Linux:  
```python3 -m venv myenv```  
```source myenv/bin/activate```
- Install the required Python libraries:  
``` pip install pillow opencv-python opencv-contrib-python numpy pymysql```
- Set up the MySQL database:
    - Create a database named authorized_user.
    - Create a table named authorized_user_table:  
```CREATE DATABASE authorized_user;```  
```USE authorized_user;```  
```CREATE TABLE authorized_user_table ( id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100),age VARCHAR(100), address VARCHAR(255);```
    - Update the database connection details in the code:  
```mydb = pymysql.connect(host="localhost", user="root", password="your_password", database="authorized_user")```
- Place the haarcascade_frontalface_default.xml file in the project directory.
- Run the application *(THIS ONE IS ENOUGH)*:  
```py '.\GUI Face Recognizer.py'```


## Usage
- *Register a User:*
    - Enter the user's name, age, and address in the respective fields.
    - Click Generate Dataset to capture 200 face images of the user.
- *Train Classifier:*
    - Click Train Classifier to train the face recognizer model with the captured dataset.
- *Detect Faces:*
    - Click Detect Faces to start real-time face recognition using the webcam.
    - The system will display recognized faces with their names and confidence levels.

## Screenshots
### Filling the details of the user:
![Filling the details of the user](https://github.com/Anurag1698/GUI-Face-Detection-Recognizer/blob/4fd0a5eeda0700b1110c2077a7c0dbef02315d60/screenshots/Fill%20Details.png)
### Generating the dataset of the user:
![Generating the dataset of the user](https://github.com/Anurag1698/GUI-Face-Detection-Recognizer/blob/4fd0a5eeda0700b1110c2077a7c0dbef02315d60/screenshots/Generating%20Dataset.png)
### Training the model on the generated dataset:
![Training the model on the generated dataset](https://github.com/Anurag1698/GUI-Face-Detection-Recognizer/blob/8e536e7fa9799d341d8c1875f6613289d373b2b2/screenshots/Training%20Dataset.png)
### Detecting the faces and displaying the name along with the confidence level:
![Detecting the faces and displaying the name along with the confidence level](https://github.com/Anurag1698/GUI-Face-Detection-Recognizer/blob/8e536e7fa9799d341d8c1875f6613289d373b2b2/screenshots/Detection1.png)
![Detecting the faces and displaying the name along with the confidence level](https://github.com/Anurag1698/GUI-Face-Detection-Recognizer/blob/8e536e7fa9799d341d8c1875f6613289d373b2b2/screenshots/Detection2.png)

## Notes
- Ensure your MySQL server is running and accessible with the credentials specified in the code.
- The captured face images are stored in the data folder, and the trained classifier is saved as classifier.xml.
- To reset the system, delete the data folder, data from the database and classifier.xml.



## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
