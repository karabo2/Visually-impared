"""
@misc{MobilNet_SSD_opencv,
  title={MobilNet_SSD for object detection on Keras and TensorFlow},
  author={djmv},
  year={2018},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/djmv/MobilNet_SSD_opencv.git}},
}
"""

#Import the neccesary libraries
import numpy as np
import argparse
import cv2 
#from gtts import gTTS
#import os
#import speech_recognition as sr
#from playsound import playsound

# construct the argument parse 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
     help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
     help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
     help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
     help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# Labels of Network.
classNames = { 'background':0,
    'aeroplane':1, 'bicycle':2 , 'bird': 3, 'boat':4,
    'bottle':5, 'bus':6, 'car':7, 'cat':8, 'chair':9,
    'cow':10, 'diningtable':11,  'dog':12,  'horse':13,
    'motorbike':14, 'person':15, 'pottedplant':16,
    'sheep':17, 'sofa':18, 'train':20,  'tvmonitor':20 }
    
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])


# Open video file or capture device. 
if args['video']:
    cap = cv2.VideoCapture(args['video'])
else:
    cap = cv2.VideoCapture(0)

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# Initialize recognizer class (for recognizing the speech)
#r = sr.Recognizer()

find='person'

find_id=classNames[find]
# Reading Microphone as source
# listening the speech and store in audio_text variable

'''
def talk():
    with sr.Microphone() as source:
        what="what are you loking for"
        file = "what.mp3"
        # initialize tts, create mp3 and play
        #tts = gTTS(what)
        #tts.save(file)
        os.system("mpg123 " + file)
        playsound(file)
        
        audio_text = r.listen(source)
        print("Time over, thanks")
    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:
            # using google speech recognition
            text=r.recognize_google(audio_text, language = 'en')
            print(text)
            if text in classes:
                scan="Scan the room"
                file = "scan.mp3"
                # initialize tts, create mp3 and play
                tts = gTTS(scan)
                tts.save(file)
                os.system("mpg123 " + file)
                playsound(file)
                
            else:
                err2='sorry I was not trained'
                file = "err1.mp3"
                # initialize tts, create mp3 and play
                #tts = gTTS(err2)
                #tts.save(file)
                os.system("mpg123 " + file)
                playsound(file)
        except:
                err2="sorry i did not get that"
                file = "err2.mp3"
                # initialize tts, create mp3 and play
                #tts = gTTS(err2)
                #tts.save(file)
                os.system("mpg123 " + file)
                playsound(file)
'''

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size. 
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300)
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob 
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()

    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .


    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction
    
        if confidence > args['confidence']: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label
            if  class_id == find_id:

                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                #CENTRE
                xcentre=int((xRightTop+xLeftBottom)/2) 
                ycentre=int((yRightTop+yLeftBottom)/2)
                
                # Draw location of central square
                x_b= int((frame_resized.shape[0]/2)-20) 
                x_t= int((frame_resized.shape[0]/2)+20)
                y_r= int((frame_resized.shape[0]/2)+20)
                y_l= int((frame_resized.shape[0]/2)-20)

                frame_resized= cv2.rectangle(frame_resized, (x_t, y_l), (x_b, y_r),(0, 255, 0))

                # Draw location of object
                frame_resized= cv2.rectangle(frame_resized, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),(0, 255, 0))

                #direction
                if xcentre > x_t:
                    
                    print('go right')

                elif xcentre < x_b:
                    
                    print('go left')

                elif ycentre < y_l:
                    
                    print('go up')

                elif ycentre > y_r:
                    
                    print('go down')

                else:
                   
                    print('stop')
                #center
                caption = '{}'.format('c')
                frame_resized= cv2.putText(
                frame_resized, caption, (xcentre, ycentre), cv2.FONT_HERSHEY_COMPLEX, 0.3, 2
            )
               
                label = find + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                frame_resized=cv2.putText(frame_resized, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

                print(label)
                break #print class and confidence
                
            
                               
    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame_resized)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break