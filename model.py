import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys


import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

from PyQt5.QtCore import QSize, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import  (
  QApplication,
  QComboBox,
  QHBoxLayout,
  QCheckBox,
  QSpinBox,
  QLabel,
  QMainWindow,
  QPushButton,
  QStackedLayout,
  QVBoxLayout,
  QWidget, 
  QStackedWidget,
)
#Mediapipe hand recognition
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

#Load the gesture recognition model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()


#Select the gestures that would not be accepted in the system of EMG
values_not_allowed = [ 7, 5, 2, 3, 9]
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('EMG and Camera data union')


        self.cap = cv2.VideoCapture(0)


        #Location from where the app starts
        self.left=30
        self.top=50

        #Size in which the app starts
        self.width=960
        self.height=520
        #Setting of the Location and Size parameters for initiation
        self.setGeometry(self.left,self.top,self.width,self.height)

        #Define the stacked widget where all the different aplication would be put on
        self.stacked_widget = QHBoxLayout(self)
        #Stack to put all the plots of the EMG data
        self.stack1 = QWidget()
        #Stack to put Camara frames
        self.stack2 = QWidget()

        self.stacked_widget.addWidget(self.stack1)
        self.stacked_widget.addWidget(self.stack2)
        #Initiate plots
        self.stack_1()

        #Initiate camera viewer

        self.VBL = QVBoxLayout()

        self.CameraLabel = QLabel()
        self.VBL.addWidget(self.CameraLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.stack2.setLayout(self.VBL)
    

        widget = QWidget()
        widget.setLayout(self.stacked_widget)
        self.setCentralWidget(widget)

    def ImageUpdateSlot(self, Image):
        self.CameraLabel.setPixmap(QPixmap.fromImage(Image))
    
    def CancelFeed(self):
        self.Worker1.stop()

    def stack_1(self):
        #Define the plot leyout box
        plots_layout = QVBoxLayout()
        #Define the figure, canva and toolbar
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        #Define the plots variables, such as axis and that sort of stuff
        self.ax1 = self.figure.add_subplot(4,2,1)
        self.sc1 = self.ax1.scatter([],[])

        self.ax2 = self.figure.add_subplot(4,2,2)
        self.sc2 = self.ax2.scatter([],[])

        self.ax3 = self.figure.add_subplot(4,2,3)
        self.sc3 = self.ax3.scatter([],[])

        self.ax4 = self.figure.add_subplot(4,2,4)
        self.sc4 = self.ax4.scatter([],[])
        
        self.ax5 = self.figure.add_subplot(4,2,5)
        self.sc5 = self.ax5.scatter([],[])
        
        self.ax6 = self.figure.add_subplot(4,2,6)
        self.sc6 = self.ax6.scatter([],[])

        self.ax7 = self.figure.add_subplot(4,2,7)
        self.sc7 = self.ax7.scatter([],[])

        self.ax8 = self.figure.add_subplot(4,2,8)
        self.sc8 = self.ax8.scatter([],[])

        self.figure.tight_layout(pad=1.08, h_pad=0, w_pad=0)

        plots_layout.addWidget(self.canvas)

        self.stack1.setLayout(plots_layout)

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)

        while self.ThreadActive:
            ret, frame = Capture.read()
            x, y , c = frame.shape
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                result = hands.process(FlippedImage)
                if result.multi_hand_landmarks:
                        landmarks = []
                        for handslms in result.multi_hand_landmarks:
                            for lm in handslms.landmark:
                                # print(id, lm)
                                lmx = int(lm.x * x)
                                lmy = int(lm.y * y)

                                landmarks.append([lmx, lmy])

                            # Drawing landmarks on frames
                            mpDraw.draw_landmarks(FlippedImage, handslms, mpHands.HAND_CONNECTIONS)
                            prediction = model.predict([landmarks])
                            # print(prediction)
                            classID = np.argmax(prediction)
                            className = classNames[classID]

                            if classID in values_not_allowed:
                                pass
                            else:
                                print(classID,className)


                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
        Capture.release()
    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())