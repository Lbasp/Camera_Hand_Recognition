import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import argparse
import logging

import numpy as np
from collections import deque
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

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

import pyqtgraph as pg
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

"""
Definition on the OpenBCI parameters for measurment
with this it is possible to start the sesion with the openBCI, getting all the information if needed
"""
BoardShim.enable_dev_board_logger()
logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser()
# use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                    default=0)
parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                    default=0)
parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM4')
parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                    required=False, default=0)
parser.add_argument('--file', type=str, help='file', required=False, default='')
parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                    required=False, default=BoardIds.NO_BOARD)
args = parser.parse_args()

params = BrainFlowInputParams()
params.ip_port = args.ip_port
params.serial_port = args.serial_port
params.mac_address = args.mac_address
params.other_info = args.other_info
params.serial_number = args.serial_number
params.ip_address = args.ip_address
params.ip_protocol = args.ip_protocol
params.timeout = args.timeout
params.file = args.file
params.master_board = args.master_board

board_shim = BoardShim(args.board_id, params)

board_shim.prepare_session()
board_shim.start_stream(450000, args.streamer_params)

"""//////////////////////////////////////////////////"""

num_points=1000
exg_channels=np.array([1, 2, 3, 4, 5, 6, 7, 8])
sampling_rate=250

"""//////////////////////////////////////////////////"""
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
values_not_allowed = [ 7, 4, 5, 2, 3, 9]
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('EMG and Camera data union')

        #Variables for camera
        self.Capture = cv2.VideoCapture(0)
        self.seg = True

        #Variables for data collection
        self.check_gest = -1
        self.check_repetition = 0
        self.df = pd.DataFrame(columns=['Gesture_ID', 'Gesture_Name', 'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6', 'EMG_7', 'EMG_8'])

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

        self.Worker1 = QTimer(self)
        self.Worker1.timeout.connect(self.ImageEMGUpdateSlot)
        self.Worker1.start(200)

        self.stack2.setLayout(self.VBL)


        widget = QWidget()
        widget.setLayout(self.stacked_widget)
        self.setCentralWidget(widget)
    


    def stack_1(self):
        #Define the plot leyout box
        plots_layout = QVBoxLayout()
        self.plot_widget1 = pg.PlotWidget()
        self.plot_widget2 = pg.PlotWidget()
        self.plot_widget3 = pg.PlotWidget()
        self.plot_widget4 = pg.PlotWidget()
        self.plot_widget5 = pg.PlotWidget()
        self.plot_widget6 = pg.PlotWidget()
        self.plot_widget7 = pg.PlotWidget()
        self.plot_widget8 = pg.PlotWidget()
        
        self.y_data1 = deque([0]*1000, maxlen=1000)
        self.y_data2 = deque([0]*1000, maxlen=1000)
        self.y_data3 = deque([0]*1000, maxlen=1000)
        self.y_data4 = deque([0]*1000, maxlen=1000)
        self.y_data5 = deque([0]*1000, maxlen=1000)
        self.y_data6 = deque([0]*1000, maxlen=1000)
        self.y_data7 = deque([0]*1000, maxlen=1000)
        self.y_data8 = deque([0]*1000, maxlen=1000)
    
        self.data_line1 = self.plot_widget1.plot(list(self.y_data1))
        self.data_line2 = self.plot_widget2.plot(list(self.y_data2))
        self.data_line3 = self.plot_widget3.plot(list(self.y_data3))
        self.data_line4 = self.plot_widget4.plot(list(self.y_data4))
        self.data_line5 = self.plot_widget5.plot(list(self.y_data5))
        self.data_line6 = self.plot_widget6.plot(list(self.y_data6))
        self.data_line7 = self.plot_widget7.plot(list(self.y_data7))
        self.data_line8 = self.plot_widget8.plot(list(self.y_data8))

        plots_layout.addWidget(self.plot_widget1)
        plots_layout.addWidget(self.plot_widget2)
        plots_layout.addWidget(self.plot_widget3)
        plots_layout.addWidget(self.plot_widget4)
        plots_layout.addWidget(self.plot_widget5)
        plots_layout.addWidget(self.plot_widget6)
        plots_layout.addWidget(self.plot_widget7)
        plots_layout.addWidget(self.plot_widget8)
        

        #Define the figure, canva
        
        self.stack1.setLayout(plots_layout)

    def ImageEMGUpdateSlot(self):
        if self.seg:
            ret, frame = self.Capture.read()
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
                                self.check_repetition=0
                            else:
                                print(className,classID)
                                if self.check_gest==classID:
                                    if self.check_repetition==4:
                                        print("Guardado")
                                        new_row = {'Gesture_ID': [classID], 'Gesture_Name': [className], 'EMG_1': [self.y_data1], 'EMG_2': [self.y_data2], 'EMG_3': [self.y_data3], 'EMG_4': [self.y_data4], 'EMG_5': [self.y_data5], 'EMG_6': [self.y_data6], 'EMG_7': [self.y_data7], 'EMG_8': [self.y_data8]}
                                        df2=pd.DataFrame.from_dict(new_row)
                                        self.df = pd.concat([self.df,df2], ignore_index=True)

                                        self.check_repetition=0
                                    else:
                                        self.check_repetition+=1
                                else:
                                    self.check_gest=classID
                                    self.check_repetition=0
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.CameraLabel.setPixmap(QPixmap.fromImage(Pic))

        data = board_shim.get_board_data(1000)
        self.y_data1.extend(data[1])
        self.y_data2.extend(data[2])
        self.y_data3.extend(data[3])
        self.y_data4.extend(data[4])
        self.y_data5.extend(data[5])
        self.y_data6.extend(data[6])
        self.y_data7.extend(data[7])
        self.y_data8.extend(data[8])

        self.data_line1.setData(self.y_data1)
        self.data_line2.setData(self.y_data2)
        self.data_line3.setData(self.y_data3)
        self.data_line4.setData(self.y_data4)
        self.data_line5.setData(self.y_data5)
        self.data_line6.setData(self.y_data6)
        self.data_line7.setData(self.y_data7)
        self.data_line8.setData(self.y_data8)

        
    def CancelFeed(self):
        self.Capture.release()
        self.seg = False
        self.df.to_csv("Merged_data.csv")






if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())