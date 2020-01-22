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
    username = db.StringField()
    password = db.StringField()
    email = db.StringField()
    confirmed = db.StringField()

    first_name = db.StringField()
    last_name = db.StringField()

    google_id = db.StringField()
    twitter_id = db.StringField()
    canvas_id = db.StringField()
    display_name = db.StringField()

    theme = db.StringField(default="original")
    plan = db.StringField(default="free")
    beta = db.StringField(default="False")
    institution = db.StringField()

    isAdmin = db.StringField(default="false")
    config = db.DictField()

@lm.user_loader
def load_user(id):
    try:
        user = User.objects.get(id=id)
    except:
        user = None

    return user

class File(db.Document):
    owner = db.ReferenceField(User)
    filename = db.StringField(unique=False)
    created = db.DateTimeField(default=datetime.datetime.now)
    status = db.StringField()
    file_type = db.StringField()
    columns = db.ListField()
    rows = db.IntField()
    filesize = db.IntField()

class Analysis(db.Document):
    owner = db.ReferenceField(User)
    created = db.DateTimeField(default=datetime.datetime.now)
    lastrun = db.DateTimeField(default=datetime.datetime.now)
    files = db.DictField()
    name = db.StringField()
    status = db.StringField()
    laststate = db.StringField()
    type = db.StringField()

    shared = db.ListField(db.ReferenceField(User))
    to_share = db.ListField(db.StringField())

class URL(db.Document):
    key = db.StringField()
    analysis_id = db.StringField()
    file_id = db.StringField()
    question_number = db.StringField()


