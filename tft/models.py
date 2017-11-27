from flask_login import UserMixin
from tft import db, lm
import datetime

class User(db.Document, UserMixin):
    social_id = db.StringField(unique=True)
    nickname = db.StringField()
    email = db.StringField()
    #email should be unique later

@lm.user_loader
def load_user(id):
    try:
        user = User.objects.get(id=id)
    except:
        user = None

    return user
'''
class Response(db.EmbeddedDocument):
    ID = db.IntField()
    rText = db.StringField()
    length = db.IntField()
    sentences = db.IntField()
    lexicalDiversity = db.StringField()
    lexicalDensity = db.StringField()
'''

class Question(db.EmbeddedDocument):
    qNum = db.IntField()
    qTitle = db.StringField()
    qText = db.StringField()
    numResponses = db.IntField()
    modelAnswer = db.StringField()
    qSettings = db.DictField()

class Analysis(db.Document):
    filename = db.StringField(unique=True)
    owner = db.ReferenceField(User)
    created = db.DateTimeField(default=datetime.datetime.now)
    questions = db.ListField(db.EmbeddedDocumentField(Question))
    tcorpus = db.StringField()