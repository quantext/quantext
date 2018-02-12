###    Quantext Text Analysis Software
###    Copyright (C) 2017,2018  McDonald & Moskal Ltd., Dunedin, New Zealand

###    This program is free software: you can redistribute it and/or modify
###    it under the terms of the GNU General Public License as published by
###    the Free Software Foundation, either version 3 of the License, or
###    (at your option) any later version.

###    This program is distributed in the hope that it will be useful,
###    but WITHOUT ANY WARRANTY; without even the implied warranty of
###    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
###    GNU General Public License for more details.

###    You should have received a copy of the GNU General Public License
###    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from flask_login import UserMixin
from tft import db, lm
import datetime

class User(db.Document, UserMixin):
    social_id = db.StringField()
    nickname = db.StringField()
    email = db.StringField()
    theme = db.StringField(default="original")
    plan = db.StringField(default="free")
    isAdmin = db.StringField(default="false")

    def save(self, *args, **kwargs):
        if not self.theme:
            self.theme = "original"
        return super(User,self).save(*args, **kwargs)

@lm.user_loader
def load_user(id):
    try:
        user = User.objects.get(id=id)
    except:
        user = None

    return user

class Question(db.EmbeddedDocument):
    qNum = db.IntField()
    qTitle = db.StringField()
    qText = db.StringField()
    numResponses = db.IntField()
    referenceAnswers = db.ListField(db.StringField())
    categories = db.ListField(db.StringField())
    blist = db.ListField(db.StringField())
    wlist = db.ListField(db.StringField())
    qSettings = db.DictField()

class File(db.Document):
    owner = db.ReferenceField(User)
    filename = db.StringField(unique=False)
    questions = db.ListField(db.EmbeddedDocumentField(Question))
    created = db.DateTimeField(default=datetime.datetime.now)
    status = db.StringField()

class RefCorpus(db.Document):
    owner = db.ReferenceField(User)
    filename = db.StringField(unique=False)
    created = db.DateTimeField(default=datetime.datetime.now)
    status = db.StringField()

class Analysis(db.Document):
    owner = db.ReferenceField(User)
    created = db.DateTimeField(default=datetime.datetime.now)
    lastrun = db.DateTimeField(default=datetime.datetime.now)
    files = db.ListField(db.ReferenceField(File))
    refcorpus = db.ListField(db.ReferenceField(RefCorpus))
    name = db.StringField()
    status = db.StringField()
    laststate = db.StringField()