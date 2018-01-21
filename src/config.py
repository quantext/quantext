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

import os

WTF_CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'

basedir = os.path.abspath(os.path.dirname(__file__))

#set defaults for file uploads
UPLOAD_FOLDER = os.path.join(basedir, 'tmp/uploads')
CORPUS_FOLDER = os.path.join(basedir, 'tmp/corpus')
FILES_FOLDER = os.path.join(basedir, 'tft/static/files')
ALLOWED_Q_EXTENSIONS = set(['xls', 'xlsx'])
ALLOWED_T_EXTENSIONS = set(['pdf', 'txt'])
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

#Twitter credentials
TWITTER_ID = "your-twitter-id"
TWITTER_SECRET = "your-twitter-secret"

#Google credentials
GOOGLE_CLIENT_ID ="your-google-client-id"
GOOGLE_CLIENT_SECRET="your-google-client-secret"

#Mongo
#NEED TO CHANGE THESE EACH TIME FOR LOCALHOST
MONGODB_SETTINGS = {
    'db': 'tft'
}

#Basic default settings
KEYWORDS = 10
#Trigrams sets length of keyphrase as 3 instead of default 2
TRIGRAM = "Trigram"
#NG_MEASURES: LR, PMI, CHISQ, STUDT, RAW
NG_MEASURE = "LR"
BLACKLIST = []
WHITELIST = []
MODEL_ANSWER = ""

# Frequency default settings
KEY_BLACKLIST = True
ONLY_WHITELIST = False
FILTER_PUNCT = True
FILTER_NUMS = False
NORM_CONTRACTIONS = True
LCASE = True
PUNCTLIST = [',','.',':',';','(',')','\\','/','_','@','#','^']
WINDOW = 6
# CGCUTOFF: >=1
CGCUTOFF = 2

#Similarity default settings
SPELLCHECK = False
#stemming is either False or 'lemma'
STEMMING = False
NEGATIVE_REPLACER = False
SYNONYM_REPLACER = False
SIM_ALGORITHM = "word2vec"

#Readability show/hide
READABILITY_WORDS = True
READABILITY_SENTS = True
READABILITY_TTR = True
READABILITY_LD = True
READABILITY_SMOG = True
READABILITY_SIMILARITY = True

OAUTH_CREDENTIALS = {
    'twitter': {
        'id': TWITTER_ID,
        'secret': TWITTER_SECRET
    },
    'google':{
        "web":
            {
                "client_id":GOOGLE_CLIENT_ID,
                "project_id":"your-project-id",
                "auth_uri":"https://accounts.google.com/o/oauth2/auth",
                "token_uri":"https://accounts.google.com/o/oauth2/token",
                "auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs",
                "client_secret":GOOGLE_CLIENT_SECRET,
                "redirect_uris":["http://127.0.0.1:8000","http://127.0.0.1:8000/google_callback"],
                "javascript_origins":["http://127.0.0.1:8000"]
            }
    }
}