from tkinter import *
from PIL import Image, ImageTk
import cv2
import imutils
import numpy as np

def visualize():
    global window
    if cap is not None:
        ret, frame = cap.read()

        if ret == True:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            #Resize video
            frame = imutils.resize(frame, width=640)

            #Convert the video
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            #Show in the GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10,visualize)
        else:
            cap.release()

#Start Function
def start():
    global cap
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    visualize()
def endit():
    cap.release()
    cv2.destroyAllWindows()


#Creation of Main Window
#Window
window = Tk()
window.title("Camera and EMG recognition")
window.geometry()
window.geometry("1280x720")

#Interface
text1 = Label(window, text="Video en Tiempo Real")
text1.place(x=580, y=10)

text2 = Label(window, text="EMG Data Adquisition")
text2.place(x=1010, y=100)

text3= Label(window, text="Gesture Detection")
text3.place(x=110, y=100)

#Botons
#Initiate Video Capture
start = Button(window, text="Initiate", height="40", width="200", command=start)
start.place(x=100, y=580)
#Stop Video
end = Button(window, text="End Data Adqusition", height="40", width="200", command=endit)
end.place(x=980, y=580)

#Video
lblVideo = Label(window)
lblVideo.place(x=320,y=50)

window.mainloop()



