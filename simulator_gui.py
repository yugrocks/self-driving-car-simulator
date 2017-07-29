import tkinter
from PIL import ImageTk
from PIL import Image

#This class provides the GUI for the simulator
class Screen:
    root = None
    id = None
    pic1 = None
    windshield = None
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title("Self Driving Vehicle Simulator")
        self.root.resizable(height=False,width=False)
        self.root.iconbitmap('wheel.ico')
        widthpixels = 700
        heightpixels = 500
        self.root.geometry('{}x{}'.format(widthpixels, heightpixels))
        #The main screen
        self.windshield = tkinter.Label(self.root, height=300, width=660)
        self.windshield.pack()
        self.windshield.place(x=20,y=10)
        self.windshield.config(background='white')

        img_src=r'wheel.png'
        self.pic1 = Image.open(img_src)
        image = self.pic1.resize((130,130))
        photo = ImageTk.PhotoImage(image)
        #The steering image, on a label
        self.steering = tkinter.Label(self.root, image=photo)
        self.steering.image = photo
        self.steering.pack()
        self.steering.place(x=40,y=330)
        self.steering.config(image= photo)     

    #to display and change the image on the windshield
    def display(self, frame):
        photo2 = Image.fromarray(frame)
        photo2 = photo2.resize((660,300))
        photo2 = ImageTk.PhotoImage(photo2)
        self.windshield.config(image= photo2)
        self.windshield.image = photo2

    #To rotate the image on the steering label
    def rotate_steering(self, degree=-30):
        img=self.pic1.rotate(degree)
        image = img.resize((130,130))
        photo2 = ImageTk.PhotoImage(image)
        self.steering.config(image= photo2)
        self.steering.image = photo2
