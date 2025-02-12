import cv2
import mediapipe as mp

class HandDetector:

    def __init__(
            self, static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ):

        self.mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def get_hands(self, image, draw=True):
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )
        
        return image
    
    def get_positions(self, image, hand_no=0):   
        landmark_list = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]

            for id, landmark in enumerate(hand.landmark):

                height, width, channels = image.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, x, y])

        return landmark_list


def trial_run():
    
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Couldn't read camera frame!")
            continue
            
        frame = detector.get_hands(frame)
        landmark_list = detector.get_positions(frame)

        if len(landmark_list) != 0:
            print(landmark_list[4])         # index of the thumb -> 4
        cv2.imshow("Image", cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    trial_run()