import cv2
import face_recognition
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json

# Fetch the service account key JSON file contents
cred = credentials.Certificate('./admin.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
   	'databaseURL':"https://fifth-jigsaw-319908-default-rtdb.firebaseio.com/"

})

# As an admin, the app has access to read and write all data, regradless of Security Rules
ref = db.reference('/exam')
print(ref.get()['surya'])


faceCascade = cv2.CascadeClassifier("D:\swa_projects\cse_projects\face_attendance\haarcascade_frontalface_default.xml")
path = 'D:/swa_projects/cse_projects/face_attendance/images'
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face =face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)



cap  = cv2.VideoCapture(0)
validity = [0,True]
nametxt = "not"
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    if nametxt == "not":
        print("scanning........")
    try:
        for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            print(matches)
            
           
            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                y1,x2,y2,x1 = faceloc
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                
                if nametxt != name:
                    print("scanning.....",1)
                    
                if validity[0]>=5:
                    nametxt = name
                    validity[0] = 0
                    print("welcome "+name)
                    if str(name) in ref.get():
                        data = ref.get()[str(name)]                   
                        print("Person confirmed")
                        cv2.putText(img,"name: "+str(name), (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                        cv2.putText(img,"age: "+str(data["age"]), (x1+10,y2+20), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                else:
                    cv2.putText(img,"scanning..", (x1,y2), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


                break
            else:
                print("not found")
                validity[0] = 0
                y1,x2,y2,x1 = faceloc
                nametxt = "not"
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(255, 0, 13),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (255, 0, 13), cv2.FILLED)
                cv2.putText(img,"Not found", (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
               
                break
    except KeyboardInterrupt:
        # If there is a KeyboardInterrupt (when you press ctrl+c), exit the program and cleanup
        print("Cleaning up!")
        
    validity[0]=validity[0]+1  
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break