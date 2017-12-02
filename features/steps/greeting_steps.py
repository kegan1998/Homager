from lettuce import *
import os
from app.homage.robot.Master import Master
from app.homage.FaceImage import FaceImage
from app.homage.camera.TestCamera import TestCamera
import inspect

world.data_path = os.path.dirname(
    inspect.getfile(inspect.currentframe())) + '/../data'


@given('a robot')
def given_robot(context):
    world.camera = TestCamera()
    world.robot = Master(camera=world.camera)
    world.robot.voice.mute()


@when('it sees a face of {name}')
def when_sees(step, name):
    face = FaceImage(filename="%s/image_%s.png"
        % (world.data_path, str(name).lower()))
    world.camera.image = face.image
    world.robot.look()


@then('it says "{text}"')
def then_said(context, text):
    assert world.robot.voice.said == str(text), \
        'expect "%s", but got "%s"' % (str(text), world.robot.voice.said)


@when('it looks at {name}')
def when_look(context, name):
    face = FaceImage(filename="%s/image_%s.png"
        % (world.data_path, str(name).lower()))
    world.camera.image = face.image
    world.robot.look()
