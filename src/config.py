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
DOC_FOLDER = os.path.join(basedir, 'doc_folder')
UPLOAD_FOLDER = os.path.join(basedir, 'tmp/uploads')
PANDAS_FOLDER = os.path.join(basedir, 'tmp/pandas')
CORPUS_FOLDER = os.path.join(basedir, 'tmp/corpus')
FILES_FOLDER = os.path.join(basedir, 'tft/static/files')
ALLOWED_Q_EXTENSIONS = set(['xls', 'xlsx', 'csv'])
ALLOWED_T_EXTENSIONS = set(['pdf', 'txt'])
MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 10

CELERY_BROKER_URL='redis://localhost:6379'
CELERY_RESULT_BACKEND='redis://localhost:6379/0'

#Mongo
#NEED TO CHANGE THESE EACH TIME FOR LOCALHOST
MONGODB_SETTINGS = {
    'db': 'tft'
    #'host': 'mongodb://tft:tft@ds155587.mlab.com:55587/tft'
}

#Basic default settings
PREFILTER = True

NKEYWORDS = 10
REVERSELISTS = False

NG_MEASURE = "STUDT"
BLACKLIST = []
WHITELIST = []
STOPLIST = "nltk"
PUNCTLIST = [',','.',':',';','(',')','\\','/','_','@','#','^']
SPECIAL_STOPS = ['URL_ADDR','EMAIL_ADDR','TWITTER_HANDLE',
                 'NUMBER_IND', 'CUSTOM_PATTERN', 'AMB_HAD_WOULD',
                 'AMB_IT_APOS_S']
CUSTOM_PATTERN = r''

FILTER_STOPS = 'ngstops'
FILTER_SPECIALS = False
#ONLY_WHITELIST = False
FILTER_PUNCT = True
FILTER_NUMS = False
FILTER_URLS = False
FILTER_EMAIL = False
FILTER_TWITTER = False
FILTER_CUSTOM = False
NORM_CONTRACTIONS = True
LCASE = "lcase"

CONTENT_POS = ['NOUN','PROPN','VERB','ADJ','ADV']
INCLUDE_POS = []
EXCLUDE_POS = []
MIN_FREQ = 1

WINDOW = 2
CGCUTOFF = 2

#Similarity default settings
SPELLCHECK = False
#stemming is either False or 'lemma'
STEMMING = False
NEGATIVE_REPLACER = False
SYNONYM_REPLACER = False
SIM_ALGORITHM = "cosine_diff"
FEATURELIST = ['1gram','2gram','3gram','noun-phrase','named-entity']
READABILITY_THRESHOLD = 30
LUMINOSO = True


OAUTH_CREDENTIALS = {
    'twitter': {
        'id': 'H1qEC1ZzBLixBOjz9sZXfknxu',
        'secret': 'RLTvVNiGig9TXbraoVe8IFNg9q6uZCL4Da7IVzXGbbfrQN6nhN'
    },
    'google':{
        "web":
            {
                "client_id":"361087200465-8mlufddta1so0aoosktih0c7cfqda5q8.apps.googleusercontent.com",
                "project_id":"text-analytics-for-teachers",
                "auth_uri":"https://accounts.google.com/o/oauth2/auth",
                "token_uri":"https://accounts.google.com/o/oauth2/token",
                "auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs",
                "client_secret":"GRtwpo6bjYvnYrdnBm0qA41S",
                "redirect_uris":["http://0.0.0.0:80","http://0.0.0.0:80/google_callback"],
                "javascript_origins":["http://0.0.0.0:80"]
            }
    }
}