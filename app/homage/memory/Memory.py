#!/usr/bin/python

from elk import Elk, ElkAttribute, attr
import inspect
import os
import sqlite3
import time
from datetime import datetime

import logging as log
log.basicConfig(level=log.INFO)  # filename='webcam.log',level=log.INFO)


class Memory(Elk):
    filename = ElkAttribute(mode='rw', type=str,
        builder='_filename', lazy=True)
    storage = ElkAttribute(mode='rw',
        builder='_storage', lazy=True)
    cache = attr(mode='rw', type=dict,
        builder='_cache', lazy=True)

    def _cache(self):
        return dict()

    def _filename(self):
        return os.path.dirname(
            os.path.abspath(
                inspect.getfile(inspect.currentframe()))) \
                    + '/../../../data/memory.db'

    def _storage(self):
        print(("loading '{}' memory file".format(self.filename)))
        c = sqlite3.connect(self.filename)
        c.execute('''CREATE TABLE IF NOT EXISTS memory
             (
             date text,
             noun text,
             verb text,
             place text
             )''')
        c.commit()
        return c

    def create_cursor(self):
        return self.storage.cursor()

    def remember(self, fields):
        place = fields['place'] or ""
        noun = fields['noun'] or ""
        verb = fields['verb'] or ""

        t = datetime.now()
        if place in self.cache:
            if noun in self.cache[place]:
                if verb in self.cache[place][noun]:
                    delay = (t - self.cache[place][noun][verb]).total_seconds()
                    if delay < 60 * 5:  # 5 minutes
                        return False
            else:
                self.cache[place][noun] = dict()
        else:
            self.cache[place] = dict()
            self.cache[place][noun] = dict()

        self.cache[place][noun][verb] = t
        log.info("logging at {} {} {} {}".format(
            place, t.strftime("%d %b %Y %H:%M:%S"), noun, verb))
        c = self.create_cursor()
        record = ("date('now')", place, noun, verb)
        c.execute(
            '''INSERT INTO memory(date,place,noun,verb)
            VALUES(?,?,?,?)''', record)
        self.storage.commit()
        return True

    def close(self):
        self.storage = None

    def query(self, fields, conditions, order=None, amount=0):
        c = self.create_cursor()
        select = "SELECT"
        select += " "
        if fields:
            select += ', '.join(fields)
        else:
            select += '*'
        select += ' FROM memory'
        if conditions:
            wheres = []
            for name, value in list(conditions.items()):
                wheres.append(" %s == '%s'" % (name, value))
            select += " WHERE " + " AND ".join(wheres)
        if order:
            select += " ORDER BY " + ", ".join(conditions)
        if amount:
            select += " LIMIT " + str(amount)
        log.debug("quering " + select)
        print(("quering " + select))
        return c.execute(select)

    def query_one(self, field, conditions, order=None):
        result = self.query([field], conditions, order, 1)
        item = next(result)
        if item:
            return item[0]
        return None

if __name__ == "__main__":
    from tempfile import gettempdir
    tmp = os.path.join(gettempdir(), '.{}'.format(hex(int(time.time()))))
    os.makedirs(tmp)
    filename = tmp + '/test1.db'

    m = Memory(filename=filename)
    m.remember({
        'noun': 'Slava',
        'verb': 'test',
        'place': 'room',
    })
    m.close()
    m = Memory(filename=filename)
    result = m.query(['noun'], {'place': 'room'})
    assert result.next()[0] == 'Slava', "Slava was in the room"

    assert m.query_one('place', {'noun': 'Slava'}) == 'room', \
        "Slava was in the room"