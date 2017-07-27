import cv2
from pynput import keyboard
import threading   

#This script was used to generate data for training and test examples.
#Data for 4 different terrains was taken from 4 different videos
frame=None
indx = [0, 0, 0, 0, 0, 0, 0, 0, 0]
basedir = r"Dataset//test_set//"

def on_press(key):
    global frame, indx
    try:
        if key.char == '2':
            print('2')
            cv2.imwrite(basedir+"a//"+str(indx[2])+".jpg",frame) #save image
            indx[2]+=1
        elif key.char == '3':
            print('3')
            cv2.imwrite(basedir+"b//"+str(indx[3])+".jpg",frame) #save image
            indx[3]+=1
        elif key.char == '4':
            print('4')
            cv2.imwrite(basedir+"c//"+str(indx[4])+".jpg",frame) #save image
            indx[4]+=1
        elif key.char == '5':
            print('5')
            cv2.imwrite(basedir+"d//"+str(indx[5])+".jpg",frame) #save image
            indx[5]+=1
        elif key.char == '6':
            print('6')
            cv2.imwrite(basedir+"e//"+str(indx[6])+".jpg",frame) #save image
            indx[6]+=1
        elif key.char == '7':
            print('7')
            cv2.imwrite(basedir+"f//"+str(indx[7])+".jpg",frame) #save image
            indx[7]+=1
        elif key.char == '8':
            print('8')
            cv2.imwrite(basedir+"g//"+str(indx[8])+".jpg",frame) #save image
            indx[8]+=1
        elif key.char == '0':
            print("skipping frame")
            
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    if key == keyboard.Key.esc:
        return False

# Collect events until released
listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)

def listen():
    listener.start()
    listener.join()
    

 
threading.Thread(target=listen).start()

#begin exploring video
vc=cv2.VideoCapture(r"videos/b.mp4")

if vc.isOpened():
    rval,frame= vc.read()
else:
    rval=False
    

        
while rval:
    rval, frame = vc.read()
    cv2.imshow("Recording", frame)
    key = cv2.waitKey(0)
    if key == 27: # exit on ESC 
        break

cv2.destroyWindow("Recording")
vc=None























