
import cv2
import os
import numpy as np
import os.path
from PIL import Image
import logging as log
import yaml
from elk import Elk, attr
import types
import inspect
import datetime
import app.homage.const as const
from FaceImage import FaceImage
from recognition.Recognition import Recognition

log.basicConfig(level=log.INFO)  # filename='webcam.log',level=log.INFO)


const.declare('LOCATION', 'location')
const.declare('SIZE', 'size')
const.declare('PREDICTED', 'predicted')
const.declare('CONFIDENCE', 'confidence')
const.declare('IMAGE', 'image')


class FaceAgent(Recognition):
    video_capture = attr(mode='ro',
        builder="_video_capture", lazy=True)
    data_path = attr(mode='ro', type=str,
        builder='_data_path')
    cascade_path = attr(mode='ro', type=str,
        builder='_cascade_path')
    face_cascade = attr(mode='ro',  # type=cv2.CascadeClassifier,
        builder="_face_cascade", lazy=True)
    photos_path = attr(mode='ro', type=str,
        builder='_photos_path')
    faces_path = attr(mode='ro', type=str,
        builder='_faces_path')
    model_name = attr(mode='ro', type=str,
        default="LBPH")
#        default="Eigen")
    recognizer = attr(mode='ro',
        builder="_recognizer", lazy=True)
    db_path = attr(mode='ro', type=str,
        builder='_db_path', lazy=True)
    log_path = attr(mode='ro', type=str,
        builder='_log_path', lazy=True)
    new_faces = attr(mode='rw', type=list,
        builder='_new_faces', lazy=True)
    facebook = attr(mode='rw',
        default=None)
    size = attr(mode='ro',
        default=(100, 100))
    show_detailed = attr(mode='ro', type=bool,
        default=True)

    eye_cascade = attr(mode='ro',
        builder='_eye_cascade', lazy=True)
    eye_cascade_path = attr(mode='ro',
        builder="_eye_cascade_path", lazy=True)

    def _photos_path(self):
        return self.data_path + "/photos"

    def _faces_path(self):
        return self.data_path + "/faces"

    def _data_path(self):
        return os.path.dirname(
            os.path.abspath(
                inspect.getfile(inspect.currentframe()))) + '/../../data'

    def _log_path(self):
        return os.path.dirname(
            os.path.abspath(
                inspect.getfile(inspect.currentframe()))) + '/../../log'

    def _cascade_path(self):
        return self.data_path + "/haarcascade_frontalface_default.xml"

    def _new_faces(self):
        return []

    def _eye_cascade_path(self):
        return self.data_path + "/haarcascade_eye_default.xml"

    def _eye_cascade(self):
        filename = self.eye_cascade_path
        if os.path.isfile(filename):
            return cv2.CascadeClassifier(filename)
        else:
            raise Exception("Invalid eye cascade file path: " + filename)

    def _recognizer(self):
        recognizer = None
        if self.model_name == 'Eigen':
            recognizer = cv2.createEigenFaceRecognizer()
        elif self.model_name == 'Fisher':
            #self.size = 300,300
            recognizer = cv2.createFisherFaceRecognizer()
        elif self.model_name == 'LBPH':
            recognizer = cv2.createLBPHFaceRecognizer(threshold=200)
        else:
            raise Exception('Unknown face recognition model "%s"'
                % (self.model_name))
        return recognizer

    def _db_path(self):
        return self.data_path + '/faces.' + self.model_name + '.data'

    def __del__(self):
        self.video_capture.release()

    def _video_capture(self):
        return cv2.VideoCapture(0)

    def _face_cascade(self):
        # For face detection we will use the Haar Cascade provided by OpenCV.
        if os.path.isfile(self.cascade_path):
            return cv2.CascadeClassifier(self.cascade_path)
        else:
            raise Exception("Invalid cascade file path: " + self.cascade_path)

    def get_db_files(self, path):
        log.info("reading files in " + path)
        files = dict()
        for entry_name in os.listdir(path):
            fullname = os.path.join(path, entry_name)
            if os.path.isfile(fullname):
                entry_name = os.path.split(entry_name)[1].split(".")[0]
            if not entry_name in files:
                files[entry_name] = []
            if os.path.isdir(fullname):
                sub_files = self.get_db_files(fullname)
                flat_list = [item for sublist in list(sub_files.values())
                    for item in sublist]
                files[entry_name].extend(flat_list)
            else:
                files[entry_name].append(fullname)
        return files

    def detect_faces(self, image):
        return self.face_cascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

    def detect_eyes(self, image):
        return self.eye_cascade.detectMultiScale(
            image,
             scaleFactor=1.1,
             minNeighbors=15,
             minSize=(25, 25),
#             flags = cv2.CASCADE_SCALE_IMAGE
        )

    def load_image(self, filename):
        log.info("reading: " + filename)
        image = Image.open(filename)
        image = image.convert('L')
        image = np.array(image, 'uint8')

        face = FaceImage(image=image)
        eyes = self.detect_eyes(face.image)
        if len(eyes) > 1:
            face.align_eyes(eyes[0], eyes[1])

        image = face.image
        return image

    def compile_faces(self):
        if not os.path.exists(self.faces_path):
            os.makedirs(self.faces_path)  # py3 , exist_ok=True)
        file_names = self.get_db_files(self.photos_path)
        for name in file_names:
            index = 0
            for filename in file_names[name]:
                image = self.load_image(filename)
                for (x, y, w, h) in self.detect_faces():
                    face_image = image[y: y + h, x: x + w]
                    face_filename = self.faces_path + '/' + name + \
                        '.' + str(index) + '.png'
                    index += 1
                    log.info("writing: " + face_filename)
                    cv2.imwrite(face_filename, face_image)
                    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#                 cv2.imshow('Video', image)
#                 while True:
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break

    def load_faces(self):
        file_names = self.get_db_files(self.faces_path)
        self.facebook = dict()
        for name in file_names:
            images = []
            for filename in file_names[name]:
                images.append(self.load_image(filename))
            self.facebook[name] = images

    def load(self):
        if not os.path.isfile(self.db_path):
            return False
        log.info("reading persistency file: " + self.db_path)
        self.recognizer.load(self.db_path)
        with open(self.db_path + '.yaml', 'r') as infile:
            self.labels = yaml.load(infile)
        return True

    def save(self, override=True):
        if not os.path.isfile(self.db_path):
            log.info("writing persistency file: " + self.db_path)
            self.recognizer.save(self.db_path)
            with open(self.db_path + '.yaml', 'w') as outfile:
                yaml.dump(self.labels, outfile, default_flow_style=False)

    def reset(self):
        if os.path.isfile(self.db_path):
            os.remove(self.db_path)

    def learn(self):
        log.info("learning images")
        images_map = self.facebook
        flat_images = []
        self.labels = []
        labels = []
        for name in sorted(images_map):
            images = images_map[name]
            multi = 1
            if self.size is None:
                flat_images.extend(images)
            else:
                for image in images:
                    small_image = cv2.resize(image, self.size)
                    flat_images.append(small_image)
                    flat_images.append(cv2.flip(small_image, 1))
                multi *= 2
            labels.extend([len(self.labels)] * (len(images) * multi))

            self.labels.append(name)
        if len(flat_images) == 0:
            raise "Not enough known faces to recognize"
        self.recognizer.train(flat_images, np.array(labels))
        self.save()
        return True

    def recognize(self, image, normalized=True, aligned=True):
        "main function to recognize faces"
        fimage = FaceImage(image=image)
        image = fimage.gray()
        faces = self.detect_faces(image)
        face_records = []
        if len(faces) == 0:
            #log.info("no faces are detected")
            pass
        for (x, y, w, h) in faces:
            face_image = FaceImage(image=image[y: y + h, x: x + w])
            log.info("processing face at (%d,%d) with (%d,%d)" % (x, y, w, h))
            if normalized:
                face_image.normalize()
                if aligned:
                    eyes = self.detect_eyes(face_image.image)
                    if len(eyes) > 1:
                        face_image.align_eyes(eyes[0], eyes[1])

            #if self.size:
                #face_image.resize(self.size)
            face_image.image = cv2.resize(face_image.image, self.size)
            predicted, confidence = self.recognizer.predict(face_image.image)
            face_records.append(
                {
                    'location': (x, y),
                    'size': (w, h),
                    'predicted': self.decode_face_id(predicted),
                    'confidence': confidence,
                    'image': face_image.image
                }
            )
        return face_records

    def predict(self, face_image):
        predicted, confidence = self.recognizer.predict(face_image)
        return {
            'predicted': self.decode_face_id(predicted),
            'confidence': confidence,
        }

    def decode_face_id(self, face_id):
        if face_id < 0 or face_id >= len(self.labels):
            return 'unknown'
        return self.labels[face_id]

    def new_face(self, image):
        if len(self.new_faces) < 10:
            self.new_faces.append(image)
            log.info("adding a new face ({})".format(len(self.new_faces)))
            return
        label = 'person' + str(len(self.labels) + 1)
        log.info("detected unknown face")
        self.labels.append(label)
        for image in self.new_faces:
            filename = None
            while filename is None:
                suffix = os.urandom(16 / 2).encode('hex')
                filename = self.faces_path + '/' + label + '.' + suffix + '.png'
                if os.path.exists(filename):
                    filename = None
            log.info("storing to " + filename)
            cv2.imwrite(filename, image)
        if self.facebook:
            self.facebook[label] = self.new_faces
        else:
            self.load_faces()
        self.new_faces = []
        self.learn()

    def remember_face(self, face):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        filename = os.path.join(
            self.log_path,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_face_" + face[const.PREDICTED]
            + ".png")
        cv2.imwrite(filename, face[const.IMAGE])

    def remember_image(self, image, title=''):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if len(title) > 0:
            title = '_image_' + title
        filename = os.path.join(
            self.log_path,
            datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            + title
            + ".png")
        cv2.imwrite(filename, image)

    def init(self):
        if self.load():
            return True
        if not os.path.isdir(self.faces_path):
            self.compile_faces()
        if not self.load():
            self.load_faces()
            self.learn()

    def run_recognition(self):
        log.info("running live recognition")
        while True:
            ret, frame = self.video_capture.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = a.recognize(image)
            self.show_faces(image, faces)

    def show_faces(self, image, faces, delay=300):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for face in faces:
            location = face['location']
            size = face['size']
            cv2.rectangle(image, location,
                (location[0] + size[0],
                location[1] + size[1]), (0, 255, 0), 2)
            if face['confidence'] > 1000:
                #self.new_face(face['image'])
                pass
            else:
                cv2.putText(
                    image,
                    face['predicted'] + ':' + str(face['confidence']),
                    (location[0], location[1] + size[1]),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.CV_AA)
        cv2.imshow('FaceAgent', image)

    def merge_faces(self, from_name, to_name):
        pass

if __name__ == "__main__":
    a = FaceAgent()
    a.init()
#     a.save()
    a.run_recognition()

    from datetime import datetime
    f = FaceImage(filename=a.photos_path + '/nick.4.jpg')
    while (1):
        startTime = datetime.now()
        a.recognize(f.image)
        timeElapsed = datetime.now() - startTime
        print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))

    faces = a.detect_faces(f.image)
    for (x, y, w, h) in faces:
#        cv2.rectangle(f.image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = f.image[y:y + h, x:x + w]

        eyes = a.detect_eyes(roi_gray)
        for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_gray,
                    (ex, ey),
                    (ex + ew, ey + eh),
                    (0, 255, 0), 2)
        result = f.align_eyes(eyes[0], eyes[1], roi_gray)
        cv2.imshow('result', result)
#        cv2.imshow('image', image)
#        cv2.imshow('new', roi_gray)
        k = cv2.waitKey(30000) & 0xff
#     if k == 27:
#         break
