###    Quantext Text Analysis Software
###    Copyright (C) 2017  McDonald & Moskal Ltd., Dunedin, New Zealand

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
ALLOWED_Q_EXTENSIONS = set(['xls', 'xlsx'])
ALLOWED_T_EXTENSIONS = set(['pdf', 'txt'])
MAX_CONTENT_LENGTH = 1 * 1024 * 1024

#Mongo
#NEED TO CHANGE THESE EACH TIME FOR LOCALHOST
MONGODB_SETTINGS = {
    'db': 'tft'
    #'host': 'mongodb://tft:tft@ds123361.mlab.com:23361/tft'
}

#Basic default settings
KEYWORDS = 10
KEYPAIRWINDOW = 2
BG_MEASURE = "RAW"
BLACKLIST = []
WHITELIST = []

# Frequency default settings
KEY_BLACKLIST = True
KEYPAIRS_BLACKLIST = False
ONLY_WHITELIST = False
FILTER_PUNCT = False
FILTER_NUMS = False
NORM_CONTRACTIONS = False
    
#Similarity default settings
SPELLCHECK = False
STEMMING = False
NEGATIVE_REPLACER = False
SYNONYM_REPLACER = False
SIM_ALGORITHM = "word2vec"

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
                "redirect_uris":["http://localhost:5000","http://localhost:5000/google_callback"],
                "javascript_origins":["http://localhost:5000"]
            }
    }
}

#OpenID not used - use OAuth instead
#OPENID_PROVIDERS = [
#    {'name': 'Google', 'url': 'https://www.google.com/accounts/o8/id'},
#    {'name': 'Yahoo', 'url': 'https://me.yahoo.com'},
#    {'name': 'AOL', 'url': 'http://openid.aol.com/<username>'},
#    {'name': 'Flickr', 'url': 'http://www.flickr.com/<username>'},
#    {'name': 'MyOpenID', 'url': 'https://www.myopenid.com'}]
