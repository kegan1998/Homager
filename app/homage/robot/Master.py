
from app.homage.FaceAgent import *
from app.homage.Voice import *
from elk import Elk, ElkAttribute
from app.homage.camera import Camera

class Master(Elk):
    face_agent = ElkAttribute(mode='rw', type=FaceAgent,
        builder="_face_agent", lazy=True) 
    voice = ElkAttribute(mode='rw', type=Voice,
        builder="_voice", lazy=True) 
    camera =ElkAttribute(mode='rw', type=Camera,
        builder="_camera", lazy=True) 
    
    def _face_agent(self):
        agent = FaceAgent()
        agent.init()
        return agent

    def _voice(self):
        return Voice()

    def _camera(self):
        return Camera()

    def best_recognized_face(self,faces):
        if not faces:
            log.info('could not find any face')
        best = None
        for face in faces:
            log.info('found face %s with %d confidence'%(face['predicted'],face['confidence']))
            if best is None or best['confidence']<face['confidence']:
                best = face
        return best

    def see(self,image):
        assert image is not None, 'image instance is mandatory'
        faces = self.face_agent.recognize(image)
        #self.face_agent.show_faces(image,faces,5000)
        face = self.best_recognized_face(faces)
        if face:
            if face['confidence']<150:
                self.voice.say('hi ' + face['predicted'])
            else:
                self.voice.say('hi')

    def look(self):
        self.see( self.camera.capture() )

if __name__ == "__main__":
    m = Master()
    print(m)
    print(m.face_agent)
    print(m.voice)
    