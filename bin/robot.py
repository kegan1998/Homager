#!/usr/bin/python

import os
import sys
import inspect

root = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe()))) + '/..'

sys.path.append(root)

import signal

from app.homage.robot.Master import Master
m = Master(
    #enable_video=True,
    voice=None
)


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    m.stop()
    #sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

m.run()
