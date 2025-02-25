import cv2
from hand_detector import HandDetector
import mediapipe as mp
import math
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

min_volume, max_volume, volume_step = volume.GetVolumeRange()
curr_volume = 0
volume_bar = 400
volume_percentage = 0

detector = HandDetector()
capture = cv2.VideoCapture(0)

while capture.isOpened():
    
    ret, frame = capture.read()
    image = detector.get_hands(frame)
    landmark_list = detector.get_positions(image)

    if len(landmark_list) != 0:

        thumb_x, thumb_y = landmark_list[4][1], landmark_list[4][2]
        index_x, index_y = landmark_list[8][1], landmark_list[8][2]
        center_x, center_y = (index_x + thumb_x) // 2, (index_y + thumb_y) // 2

        cv2.circle(image, (thumb_x,thumb_y), 13, (216,47,95), -1)
        cv2.circle(image, (index_x,index_y), 13, (216,47,95), -1)
        cv2.line(image, (thumb_x,thumb_y), (index_x,index_y), (216,47,95),3)

        length = math.hypot(index_x-thumb_x, index_y-thumb_y)
        if length < 50:
            cv2.circle(image, (center_x,center_y), 15, (0,255,0), -1)

        curr_volume = np.interp(length, [0,180], [min_volume,max_volume])
        volume_bar = np.interp(length, [40,200], [400,150])
        volume_percentage = np.interp(length, [40,200], [0,100])
        volume.SetMasterVolumeLevel(curr_volume, None)

    cv2.rectangle(image, (50,150), (85,400), (0,0,0), 3)
    cv2.rectangle(image, (50,int(volume_bar)), (85,400), (47,121,239), cv2.FILLED)
    cv2.putText(image, f'{int(volume_percentage)} %', (40,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,204), 2)

    cv2.imshow("Capture", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()