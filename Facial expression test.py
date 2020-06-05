from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Users\Cooper\Desktop\Face Expression\assests\haarcascade_frontalface_default.xml')#path to harcasde classifer
model = load_model(r'C:\Users\Cooper\Desktop\Face Expression\assests\Emotion_model.h5') #path to emotion model

class_labels = ['Angry','Happy','Sad','Neutral','Surprise']

cap = cv2.VideoCapture(0) #0 for web cam and 1 for external camera 

while True:
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
       
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            preds = model.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x,y)
            thickness=1;
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness,3)
            
        else:
            
            cv2.putText(frame,'No Face Detection',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),thickness,3)


    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
                
                   
