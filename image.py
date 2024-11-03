import cv2    
import time
cpt = 0 # for assigning name to each img from video
maxFrames = 100 # no of images

count=0 # counter for images till now 

cap=cv2.VideoCapture('video\mycarplate.mp4') # video file path for capturing images 
while (cpt < maxFrames):
    ret, frame = cap.read() # ret: boolean if frame captured or not successfully, frame:represents numpy array if ret is true, if ret is null represents nothing 

    if not ret:  # if ret: false then break the loop  
        break   

    count += 1  # count the frame 
    if count % 3 != 0:  # processed only third frame 
        continue
    frame=cv2.resize(frame,(1080,500)) # resize the frame 
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"C:\Users\HP\Desktop\ANPR1\test_images\numberplate_%d.jpg" %cpt, frame) # store in that folder 
    time.sleep(0.01) # wait for 0.01 second
    cpt += 1    # increase the counter 
    if cv2.waitKey(5)&0xFF==27:  # stop when `ESC` is pressed 
        break
cap.release()   # when while loop ends release the video
cv2.destroyAllWindows() # destroy all the windows 