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

#Twitter credentials
TWITTER_ID = "your-twitter-id"
TWITTER_SECRET = "your-twitter-secret"

#Google credentials
GOOGLE_CLIENT_ID ="your-google-client-id"
GOOGLE_CLIENT_SECRET="your-google-client-secret"

#Mongo
MONGODB_SETTINGS = {
    'db': 'tft'
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
        'id': TWITTER_ID,
        'secret': TWITTER_SECRET
    },
    'google':{
        "web":
            {
                "client_id":GOOGLE_CLIENT_ID,
                "project_id":"Quantext",
                "auth_uri":"https://accounts.google.com/o/oauth2/auth",
                "token_uri":"https://accounts.google.com/o/oauth2/token",
                "auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs",
                "client_secret":GOOGLE_CLIENT_SECRET,
                "redirect_uris":["http://localhost:8000","http://localhost:8000/google_callback"],
                "javascript_origins":["http://localhost:8000"]
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
