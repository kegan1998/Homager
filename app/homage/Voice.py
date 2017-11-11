
from elk import Elk, ElkAttribute
import types
import logging as log

log.basicConfig(level=log.INFO)#filename='webcam.log',level=log.INFO)

import requests
requests.packages.urllib3.disable_warnings()

from gtts import gTTS
import os
import pygame as pg
# pip install mutagen
import mutagen.mp3

class Voice(Elk):
    said = ElkAttribute(mode='rw', type=str)

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
            # check if playback has finished
            clock.tick(30)
    
    def say(self,text):
        self.said = text
        tts = gTTS(text=text, lang='en')
        filename = "pcvoice.mp3"
        tts.save(filename)
        self.play_file(filename)
#         if os.name == 'nt':
#             os.startfile(filename)
#         else:
#             os.system("mpg321 " + filename)

if __name__ == "__main__":
    v = Voice()
    v.say("Hello Nick, come here. What's up?")
