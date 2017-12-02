
from elk import Elk, ElkAttribute
import types
import logging as log

log.basicConfig(level=log.INFO)#filename='webcam.log',level=log.INFO)

import requests
requests.packages.urllib3.disable_warnings()

from gtts import gTTS
import os
import pygame as pg
import mutagen.mp3
from tempfile import gettempdir

class Voice(Elk):
    said = ElkAttribute(mode='rw', type=str)
    muted = ElkAttribute(mode='rw', type=bool,
        default=False)
    temp_path = ElkAttribute(mode='rw', type=str,
        builder='_temp_path',lazy=True)
    play_filename = ElkAttribute(mode='rw', type=str,
        default='pcvoice.mp3')
    
    def _temp_path(self):
        path = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        os.makedirs(path)
        return path

    def mute(self):
        self.muted = True

    def unmute(self):
        self.muted = False

    def play_file(self, music_file, volume=0.8):
        # set up the mixer    
#         freq = 44100     # audio CD quality
#         bitsize = -16    # unsigned 16 bit
#         channels = 1     # 1 is mono, 2 is stereo
#         buffer = 2048    # number of samples (experiment to get best sound)
#         buffer = 4096  # number of samples (experiment to get best sound)
#         pg.mixer.init(freq, bitsize, channels, buffer)

        mp3 = mutagen.mp3.MP3(music_file)
        pg.mixer.init(frequency=mp3.info.sample_rate)
    
        # volume value 0.0 to 1.0
        pg.mixer.music.set_volume(volume)
        clock = pg.time.Clock()
        try:
            pg.mixer.music.load(music_file)
        except pg.error:
            log.error("File {} not found! ({})".format(music_file, pg.get_error()))
            return
        pg.mixer.music.play()
        while pg.mixer.music.get_busy():
            clock.tick(30)
        pg.mixer.music.stop()
        pg.mixer.quit()
    
    def say(self,text):
        self.said = text
        if not self.muted:
            tts = gTTS(text=text, lang='en')
            tts.save(self.play_filename)
            self.play_file(self.play_filename)

if __name__ == "__main__":
    v = Voice()
    v.say("Hello Nick, come here. What's up?")
