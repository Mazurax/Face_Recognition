import face_recognition


class ImageToArray:

    def __init__(self, res_frame, box):
        (startX, startY, endX, endY) = box.astype("int")
        self.crop = res_frame[startX:endX, startY:endY]

    def get_array(self):
        print  face_recognition.face_encodings(self.crop)
