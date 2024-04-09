import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from random import randint as ri
import os
from pygame import mixer

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("emotion_model.h5")

cap = cv2.VideoCapture('video3.mp4') #Input the video path/ '0' to read from webcam
emotion_array=[]

while True:
    ret, frame = cap.read() 
    frame = cv2.resize(frame, (800, 500))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        print("Your mood is detected")
        print(emotion_dict[maxindex])
        cv2.imwrite('Emotion Detection.jpg', frame)
        cap.release()

        emotion_array.append(emotion_dict[maxindex])
        prevailing_emotion = emotion_array[0]
        for i in emotion_array:
            if emotion_array.count(i) > emotion_array.count(prevailing_emotion):
                prevailing_emotion = i

        prevailing_emotion = prevailing_emotion.lower()
        emo_song_dict = {"angry": "calm", "disgusted": "energetic", "fearful": "calm", "happy": "happy",
                         "neutral": "happy", "sad": "happy", "surprised": "energetic"}
        song_no = ri(1, 2)

        absolutepath = os.path.abspath(__file__)
        fileDirectory = os.path.dirname(absolutepath)
        
        fileDirectory = os.path.join(fileDirectory, 'data') #create a folder named data
        fileDirectory = os.path.join(fileDirectory, 'songs') #create a folder named 'songs' indide 'data', inside which create multiple folders with name of emotions and add some songs in it
        fileDirectory = os.path.join(fileDirectory, emo_song_dict[prevailing_emotion])

        mixer.init()
        mixer.music.load(fileDirectory + '\\0' + str(song_no) + '.mp3')
        mixer.music.play()
        print("Music playing....")

        command = input("To stop the music please type,'stop':")
        if command == 'stop':
            mixer.music.stop()
        print("END")
        break

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

