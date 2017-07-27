import cv2
from keras.models import model_from_json
from functools import partial
from simulator_gui import Screen
import threading


#specifying image dimensions
img_rows=33
img_colms=50
img_channels=1 #1 for grayscale, and 3 for RGB images 

def load_model():
    #load the model
    try:
        model_file = open('model2.json', 'r')
        loaded_model = model_file.read()
        model_file.close()
        model = model_from_json(loaded_model)
        #load weights into the model
        model.load_weights("weights2.hdf5")
        print("Model loaded successfully")
        return model
    except:
        print("Error while loading model. Terminating")
        return None


def start_simulation(path, screen):
    #To start the main simulation
    model = load_model() #load model
    if model == None:
        quit()
    print("Model Loaded Successfully. Starting simulator.")
    
    classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g'] #defining classes
    degrees=[30, 20, 10, 0, -10, -20, -30] #defining the steering rotations corresponding to classes

    vc=cv2.VideoCapture(path) 
    if vc.isOpened():
        rval,frame= vc.read() #capture the first frame
    else:
        rval=False

    while rval:
        rval, img = vc.read()
        if path==0:
            img=cv2.flip(img,1)
        screen.display(img)  #display the frame on screen
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        img = cv2.resize(img, (img_colms, img_rows), interpolation = cv2.INTER_AREA)
        img=img.reshape((1,)+img.shape+(1,))
        y_pred = model.predict_classes(img, batch_size=1)[0] #predict its class
        screen.rotate_steering(degree=degrees[y_pred])  #rotate the steering accordingly
        cls = classes[y_pred]
        print(cls)
    vc=None


def main():
    
    a = str(input("""Choose:\n
        1 - Run simulator on a video from disk.
        2 - Capture a video from webcam and play the simulator.
        """))    
    
    if a == '1':  #start simulation on a pre existing video on disk
        path = input("Please input the full path of the video in the disk\n")
        screen = Screen() #get the  GUI
        threading.Thread(target=partial(start_simulation,path,screen)).start() #start simulation in another thread    
    elif a== '2':   #use the webcam to capture real scene
        screen = Screen() 
        threading.Thread(target=partial(start_simulation,path,screen)).start()
    else:
        print("Wrong choice. Terminating.")
        quit()
    screen.root.mainloop() #The GUI mainloop

#begin
main()