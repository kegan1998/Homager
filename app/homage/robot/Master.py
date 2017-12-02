
from app.homage.FaceAgent import *
from app.homage.Voice import *
from elk import Elk, ElkAttribute, attr
from app.homage.camera.Camera import Camera
from app.homage.camera.RPICamera import RPICamera
from app.homage.memory.Memory import Memory

from types import NoneType
import time


class Master(Elk):
    face_agent = ElkAttribute(mode='rw', type=FaceAgent,
        builder="_face_agent", lazy=True)
    voice = ElkAttribute(mode='rw', type=(Voice, NoneType),
        builder="_voice", lazy=True)
    saw = ElkAttribute(mode='rw', type=str,
        default='')
    face_recognition_threshold = ElkAttribute(mode='rw', type=int,
        default=150)
    camera = ElkAttribute(mode='rw', type=Camera,
        builder="_camera", lazy=True)
    memory = attr(mode='rw', type=Memory,
        builder="_memory", lazy=True)
    enable_video = ElkAttribute(mode='rw', type=bool,
        default=False)
    sleep_delay = ElkAttribute(mode='rw', type=float,
        default=1.0)
    interrupt = attr(mode='rw', type=bool,
        default=False)

    def _face_agent(self):
        agent = FaceAgent()
        agent.init()
        return agent

    def _voice(self):
        return Voice()

    def _camera(self):
        return RPICamera()

    def _memory(self):
        return Memory()

    def best_recognized_face(self, faces):
        best = None
        index = 0
        for face in faces:
            print_confidence = face['confidence']
            if print_confidence > 1000:
                print_confidence = 'uncertain'
            else:
                print_confidence = str(print_confidence)
            if self.enable_video:
                index += 1
                cv2.imshow('Face' + str(index), face['image'])

            log.info('found face %s with %s confidence' %
                (face['predicted'], print_confidence))
            if best is None or best['confidence'] < face['confidence']:
                best = face
        return best

    def biggest_face(self, faces):
        biggest = None
        biggest_size = 0
        for face in faces:
            size = face['size']
            size = (size[0] + size[1]) / 2
            if size > biggest_size:
                biggest = face
                biggest_size = size
        return biggest

    def recognize(self, image):
        assert image is not None, 'image instance is mandatory'
        self.faces = self.face_agent.recognize(image)
        if self.enable_video:
            self.face_agent.show_faces(image, self.faces, 5000)
            pass
        saw = ''
        self.face = None
        self.biggest_face = None
        if not self.faces:
            #log.info("could not find any face")
            pass
        else:
            self.face = self.best_recognized_face(self.faces)
            if not self.face:
                #log.info("could not recognize any face")
                saw = 'FACE'
            else:
                if self.face['confidence'] < self.face_recognition_threshold:
                    saw = self.face['predicted']
                else:
                    saw = 'FACE'
            if saw == 'FACE':
                self.biggest_face = self.biggest_face(faces)
                if self.biggest_face:
                    log.info("found a new face")
                    self.face_agent.new_face(self.biggest_face['image'])
        self.saw = saw

    def look(self):
        image = self.camera.capture()
        self.recognize(image)
        if self.saw == 'FACE':
            self.voice.say('hi, I do not know you')
        elif self.saw != '':
            if self.voice:
                self.voice.say('hi ' + self.saw)
            if self.memory:
                if self.memory.remember({
                        'place': 'kitchen',
                        'noun': self.saw,
                        'verb': 'see'
                    }):
                    self.face_agent.remember_face(self.face)
                    self.face_agent.remember_image(image, self.saw)

    def stop(self):
        self.interrupt = True

    def run(self):
        while not self.interrupt:
            self.look()
            if self.enable_video:
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break
            else:
                if self.saw == '':
                    if self.sleep_delay > 3.0:
                        self.sleep_delay -= 0.1
                    else:
                        self.sleep_delay = 3.0
                else:
                    self.sleep_delay = 5.0
                time.sleep(self.sleep_delay)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    m = Master(
        enable_video=True
    )
    m.run()
