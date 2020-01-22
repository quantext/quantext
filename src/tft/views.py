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

from flask import render_template, flash, url_for, redirect, request, jsonify, session, stream_with_context, send_from_directory, send_file
from flask import Response as flask_response
from tft import app, db, celery

from copy import copy
import chardet
import tempfile

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

import requests
import itertools

from werkzeug import secure_filename
from flask_login import login_user, logout_user, current_user

from oauth import OAuthSignIn
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from .models import User, Analysis, File, URL
import datetime
import nlputility

import spacy
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc

from ftfy import fix_encoding
from cytoolz import itertoolz
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk import FreqDist
import lexical_diversity as ldiv
import pyphen
from math import sqrt, floor, ceil, isnan

import numpy as np

from collections import defaultdict
import json
import string
import random
import os
import re
import textract

import time
import redis
import pickle

#Tokenizer settings - suggest leave this out of interface for now!
PREFIX = r'''^[\[\("']'''
INFIX = r'''[~]'''
SUFFIX = r'''[\]\)"',.:;!?]$'''

prefix_re = re.compile(PREFIX)
suffix_re = re.compile(SUFFIX)
infix_re = re.compile(INFIX)

nltk_stops = []
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'stopwords.txt'), 'r', encoding='utf8') as f:
    stopsbyline = f.read()
    nltk_stops = stopsbyline.splitlines()

def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, rules={},
                     prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer
                     )

nlp = spacy.load('en_core_web_md')
nlp.tokenizer = custom_tokenizer(nlp)
spacy_stops = list(nlp.Defaults.stop_words)

def fix_unicode(text):
    return fix_encoding(text)

def strip_whitespace(text):
    return ' '.join(text.split())

r = redis.Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

# initialise session vars - not sure this is at all secure!! Need to check :|
def default_settings():
    question_settings = {}
    question_settings.update({'nkey':app.config['NKEYWORDS']})
    question_settings.update({'reverse_lists':app.config['REVERSELISTS']})

    question_settings.update({'punctuation':app.config['PUNCTLIST']})
    question_settings.update({'specialstops':app.config['SPECIAL_STOPS']})
    question_settings.update({'custom':app.config['CUSTOM_PATTERN']})
    question_settings.update({'stoplist':app.config['STOPLIST']})
    question_settings.update({'blacklist':app.config['BLACKLIST']})
    question_settings.update({'whitelist':app.config['WHITELIST']})

    question_settings.update({'filter_stops':app.config['FILTER_STOPS']})
    question_settings.update({'filter_specials':app.config['FILTER_SPECIALS']})
    question_settings.update({'filter_punct':app.config['FILTER_PUNCT']})
    question_settings.update({'filter_urls':app.config['FILTER_URLS']})
    question_settings.update({'filter_email':app.config['FILTER_EMAIL']})
    question_settings.update({'filter_twitter':app.config['FILTER_TWITTER']})
    question_settings.update({'filter_nums':app.config['FILTER_NUMS']})
    question_settings.update({'filter_custom':app.config['FILTER_CUSTOM']})

    question_settings.update({'content_pos':app.config['CONTENT_POS']})
    question_settings.update({'include_pos':app.config['INCLUDE_POS']})
    question_settings.update({'exclude_pos':app.config['EXCLUDE_POS']})
    question_settings.update({'min_freq':app.config['MIN_FREQ']})

    question_settings.update({'measure':app.config['NG_MEASURE']})
    question_settings.update({'modelanswer':""})

    question_settings.update({'norm_contractions':app.config['NORM_CONTRACTIONS']})
    question_settings.update({'lcase':app.config['LCASE']})
    question_settings.update({'window':app.config['WINDOW']})
    question_settings.update({'cgcutoff':app.config['CGCUTOFF']})

    question_settings.update({'simalgorithm':app.config['SIM_ALGORITHM']})
    question_settings.update({'featurelist':app.config['FEATURELIST']})
    question_settings.update({'readability_threshold':app.config['READABILITY_THRESHOLD']})
    question_settings.update({'luminoso':app.config['LUMINOSO']})

    question_settings.update({'randomise_labels':False})
    question_settings.update({'add_new_labels':False})
    question_settings.update({'aliases':{"Categories_1":"Default categories"}})

    question_settings.update({'show_freq':False})
    question_settings.update({'public':True})

    #question_settings.update({'spell':SPELLCHECK})
    #question_settings['negreplace'] = ['NEGATIVE_REPLACER']
    #question_settings['synreplace'] = ['SYNONYM_REPLACER']

    return question_settings

#check uploaded file has extension specified
def allowed_fileq(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_Q_EXTENSIONS']

#check uploaded file has extension specified
def allowed_filet(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_T_EXTENSIONS']

#hack function to generate random id and concatenate to filename - avoids duplicate names and
#easy tracking of issues without db query etc :)
#need to deal with files properly once authentication in place
def fname_generator(filename, size=6, chars=string.ascii_uppercase + string.digits):
    filenameparts = filename.rsplit('.', 1)
    newid = ''.join(random.choice(chars) for _ in range(size))
    return filenameparts[0] + '_' + newid + '.' + filenameparts[1]

#adapted from http://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
def sort_ids(qlist):
    """ Sort question ids whether alpha, numeric or both."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(qlist, key = alphanum_key)

def pdf_to_text(f):
    try:
        print('convertingfile... ')
        extr = textract.process(f, encoding='utf8')
        return extr.decode('utf8')
    except:
        print(f, ' conversion to text failed')

    return

def get_selected(svar, listvals):
    selectedlist = []

    for i in listvals:
        if i == svar:
            selectedlist.append((i,'selected'))
        else:
            selectedlist.append((i,None))
    return selectedlist

def process_response(response):
    response = fix_unicode(response)
    response = strip_whitespace(response)
    return response

def process_settings(question_settings):
    processed_settings = {}

    #TRUE/FALSE
    processed_settings.update({"reverse_lists":question_settings['reverse_lists'] == True or question_settings['reverse_lists'] == 'True'})
    processed_settings.update({"filter_specials":question_settings['filter_specials'] == True or question_settings['filter_specials'] == 'True'})
    processed_settings.update({"filter_punct":question_settings['filter_punct'] == True or question_settings['filter_punct'] == 'True'})
    processed_settings.update({"norm_contractions":question_settings['norm_contractions'] == True or question_settings['norm_contractions'] == 'True'})
    processed_settings.update({"filter_urls":question_settings['filter_urls'] == True or question_settings['filter_urls'] == 'True'})
    processed_settings.update({"filter_email":question_settings['filter_email'] == True or question_settings['filter_email'] == 'True'})
    processed_settings.update({"filter_twitter":question_settings['filter_twitter'] == True or question_settings['filter_twitter'] == 'True'})
    processed_settings.update({"filter_nums":question_settings['filter_nums'] == True or question_settings['filter_nums'] == 'True'})
    processed_settings.update({"filter_custom":question_settings['filter_custom'] == True or question_settings['filter_custom'] == 'True'})
    processed_settings.update({"luminoso":question_settings['luminoso'] == True or question_settings['luminoso'] == 'True'})
    processed_settings.update({"randomise_labels":question_settings['randomise_labels'] == True or question_settings['randomise_labels'] == 'True'})
    processed_settings.update({"show_freq":question_settings['show_freq'] == True or question_settings['show_freq'] == 'True'})
    processed_settings.update({"public":question_settings['public'] == True or question_settings['public'] == 'True'})

    #NUMBERS
    processed_settings.update({"min_freq":int(question_settings['min_freq'])})
    processed_settings.update({"window":int(question_settings['window'])})
    processed_settings.update({"cgcutoff":int(question_settings['cgcutoff'])})
    processed_settings.update({"nkey":int(question_settings['nkey']) if question_settings['nkey'] != "all" else None})
    processed_settings.update({"readability_threshold":int(question_settings['readability_threshold'])})

    #LISTS
    processed_settings.update({"featurelist":question_settings['featurelist']})
    processed_settings.update({"content_pos":question_settings['content_pos']})
    processed_settings.update({"include_pos":question_settings['include_pos']})
    processed_settings.update({"exclude_pos":question_settings['exclude_pos']})
    processed_settings.update({"specialstops":question_settings['specialstops']})
    processed_settings.update({"blacklist":question_settings['blacklist']})
    #take this out again for production:
    if 'whitelist' in question_settings:
        processed_settings.update({"whitelist":question_settings['whitelist']})
    else:
        processed_settings.update({"whitelist":[]})

    #STRINGS
    processed_settings.update({"lcase":question_settings['lcase']})
    processed_settings.update({"filter_stops":question_settings['filter_stops']})
    processed_settings.update({"stoplist":question_settings['stoplist']})
    processed_settings.update({"custom":question_settings['custom']})
    processed_settings.update({"measure":question_settings['measure']})

    return processed_settings

def process_doc(file_id,col_idx,row_idx,text):
    d = nlp(text)
    origpath = os.path.join(app.config['DOC_FOLDER'],file_id,col_idx,row_idx)
    d.doc.to_disk(origpath)

    return d

def process_text(type,question,search=None):
    processed_docs = process_responses(question['responses'])
    return processed_docs

def process_reference_statistics(text):
    processed_docs = process_responses([{'text':text}])
    statistics = get_processed_stats(processed_docs)
    return statistics

def iter_text(question_responses):
    for resp in question_responses:
        yield strip_whitespace(fix_unicode(resp['text']))

#all the helper functions shoved down here for now :)
def process_responses(data):
    processed_docs = nlp.pipe(iter_text(data),batch_size=500,n_threads=4)
    return list(processed_docs)

def get_stops(question_settings):
    list = []
    stoplist = question_settings['stoplist']
    if stoplist == 'nltk':
        list = nltk_stops
    elif stoplist == 'spacy':
        list = spacy_stops
    newlist = list[:]
    newlist.extend(question_settings['blacklist'])
    newlist = [word for word in newlist if word not in question_settings['whitelist']]
    return newlist

def normalise_contractions(text):
    """Adapted from textacy library"""
    # standard
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould|[Ww]as)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)
    # non-standard
    text = re.sub(r"(\b)([Nn])t", r"\1\2ot", text)
    text = re.sub(r"(\b)([Nn])'t", r"\1\2ot", text)
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
    # ambiguous
    text = re.sub(r"(\b)([Ii]t)'s", r"\1\2_s", text)
    text = re.sub(r"(\b)([Ii]|[Ww]e|[Yy]ou|[Tt]hey|[Ii]t)'d", r"\1\2_d", text)
    return text

def findtopbigrams(word_fd,bg_fd,settings):
    nkey = settings['nkey']
    metric = select_bg_measure(settings,False)
    finder = BigramCollocationFinder(word_fd, bg_fd)
    warning = ""

    try:
        top_bigrams = finder.nbest(select_bg_measure(settings,False), nkey)
    except:
        warning = "Problem with %s measure for bigrams. Default to simple frequency (RAW setting)" % metric.__name__
        print(warning)
        top_bigrams = finder.nbest(select_bg_measure(settings,True), nkey)

    #score bigrams using LR or similar measure but more helpful to end user to see raw counts + explain measure used in tool tip
    top_bg_with_count = sorted([(bg, count) for (bg, count) in finder.ngram_fd.items() if bg in top_bigrams], key=lambda bgcount:-bgcount[1])
    top_bigrams = [[list(bg), count] for (bg, count) in top_bg_with_count if count > 0 and bg[0]!=bg[1]]
    return top_bigrams,warning

def findtoptrigrams(word_fd,bg_fd,tg_fd,settings):
    nkey = settings['nkey']
    metric = select_tg_measure(settings,False)
    wild = [(k[0],k[-1]) for k in tg_fd.keys()]
    wild_fd = FreqDist(wild)
    finder = TrigramCollocationFinder(word_fd, bg_fd, wild_fd, tg_fd)
    warning = ""

    try:
        top_trigrams = finder.nbest(select_tg_measure(settings,False), nkey)
    except:
        warning = "Problem with %s measure for trigrams. Default to simple frequency (RAW setting)" % metric.__name__
        print(warning)
        top_trigrams = finder.nbest(select_tg_measure(settings,True), nkey)

    #score trigrams using LR or similar measure but more helpful to end user to see raw counts + explain measure used in tool tip
    top_tg_with_count = sorted([(tg, count) for (tg, count) in finder.ngram_fd.items() if tg in top_trigrams], key=lambda tgcount:-tgcount[1])
    top_trigrams = [[list(tg), count] for (tg, count) in top_tg_with_count if count > 0 and tg[0]!=tg[1]!=tg[-1]]
    return top_trigrams, warning

def get_words(doc):
    return list(word for word in doc if not word.is_punct and not word.is_space)

def get_pos(words):
    return [[token.text,token.pos_] for token in words]

def get_content_words(words, settings):
    content_pos = settings['content_pos']
    n_content_words = len([w for w in words if w.pos_ in content_pos])
    return n_content_words

def get_statistics(doc):
    #PROCESSING...
    words_doc = get_words(doc)
    words = [word.lower_ for word in words_doc]
    sentences = list(doc.sents)

    #COUNTS...
    n_chars = sum([len(word) for word in words])
    n_words = len(words)
    n_sents = len(sentences)

    return {
        "Characters":n_chars,
        "Words":n_words,
        "Sentences":n_sents
    }

def get_indices(doc, settings):
    #PROCESSING...
    readability_threshold = settings['readability_threshold']
    words_doc = get_words(doc)
    words = [word.lower_ for word in words_doc]
    sentences = list(doc.sents)
    unique_words = list({word for word in words})

    #COUNTS...
    n_words = len(words)
    n_unique_words = len(unique_words)
    n_sents = len(sentences)
    n_content_words = get_content_words(words_doc, settings)
    n_syllables, n_polysyllable_words = count_syllables(words,settings)

    #CALCULATIONS...
    ldiv_ttr = "{0:.2f}".format(ttr(n_words,n_unique_words,readability_threshold))
    ldiv_mtld = "{0:.2f}".format(MTLD(words))
    ldiv_hdd = "{0:.2f}".format(hdd(words))
    ld = "{0:.2f}".format(lexical_density(n_words,n_content_words,readability_threshold))
    smog = "{0:.2f}".format(smog_measure(n_words,n_polysyllable_words,n_sents,readability_threshold))
    gunning = "{0:.2f}".format(gunning_fog(n_words,n_sents,n_polysyllable_words,readability_threshold))
    flesch = "{0:.2f}".format(flesch_ease(n_words,n_syllables,n_sents,readability_threshold))
    fk = "{0:.2f}".format(flesch_kincaid(n_words,n_syllables,n_sents,readability_threshold))
    return {
        "Unique_words":unique_words,
        "TTR":ldiv_ttr,
        "MTLD":ldiv_mtld,
        "HDD":ldiv_hdd,
        "LD":ld,
        "SMOG":smog,
        "Gunning":gunning,
        "Flesch":flesch,
        "FK":fk
    }

def select_bg_measure(settings,usedefault=False):
    measure = settings['measure']
    bigram_measures = BigramAssocMeasures()

    if measure == 'RAW' or usedefault:
        return bigram_measures.raw_freq
    elif measure == 'PMI':
        return bigram_measures.pmi
    elif measure == 'LR':
        return bigram_measures.likelihood_ratio
    elif measure == 'STUDT':
        return bigram_measures.student_t
    elif measure == 'CHISQ':
        return bigram_measures.chi_sq

def select_tg_measure(settings,usedefault=False):
    measure = settings['measure']
    trigram_measures = TrigramAssocMeasures()

    if measure == 'RAW'or usedefault:
        return trigram_measures.raw_freq
    elif measure == 'PMI':
        return trigram_measures.pmi
    elif measure == 'LR':
        return trigram_measures.likelihood_ratio
    elif measure == 'STUDT':
        return trigram_measures.student_t
    elif measure == 'CHISQ':
        return trigram_measures.chi_sq

def count_features(processed_collection,settings,feat):
    feats = []
    if feat == 'np':
        feats = [list(d.noun_chunks) for d in processed_collection if type(d) is not str]
    if feat == 'ne':
        feats = [list(e for e in d.ents) for d in processed_collection if type(d) is not str]
    if feat == 'custom':
        pass
    feats = [f for ft in feats for f in ft]

    return get_fd(feats,settings)

def combine_fds(fd_dicts):
    combined = {}
    for d in fd_dicts:
        for k,v in d.items():
            if k in combined.keys():
                combined[k]+=d[k]
            else:
                combined[k] = v
    return FreqDist(combined)

def apply_filters(doc,settings):
    tokens = [t for t in doc]

    if settings['filter_punct']:
        tokens = [t for t in tokens if not t.is_punct]
    if settings['filter_nums']:
        tokens = [t for t in tokens if not t.like_num]
    if settings['filter_urls']:
        tokens = [t for t in tokens if not t.like_url]
    if settings['filter_email']:
        tokens = [t for t in tokens if not t.like_email]

    if settings['lcase']=="lcase":
        words = [t.lower_ for t in tokens]
    elif settings['lcase']=="lemma":
        words = [t.lemma_ if (t.tag_ != 'PRP$' and t.tag_ != 'PRP') else t.lower_ for t in tokens ]
    else:
        words = [t.text for t in tokens]

    if settings['norm_contractions']:
        words = [normalise_contractions(w).split() for w in words]
        words = [w for wd in words for w in wd]

    return words

def filter_stops(finder,stops):
    wf = lambda w: w in stops
    finder.apply_word_filter(wf)
    return finder.ngram_fd

def filter_tg_stops(finder,stops):
    wf = lambda w1, w2, w3: (w1) in stops or (w3) in stops
    finder.apply_ngram_filter(wf)
    return finder.ngram_fd

def get_bgcollocates_of_word(finder,word):
    wf = lambda w1,w2: word not in (w1,w2)
    finder.apply_ngram_filter(wf)
    return finder.ngram_fd

def get_tgcollocates_of_word(finder,word):
    wf = lambda w1,w2,w3: word not in (w1,w2,w3)
    finder.apply_ngram_filter(wf)
    return finder.ngram_fd

def get_features(processed_collection,settings,word=None):
    window = settings['window']
    tgwindow = 3 if window == 2 else window

    no_stops = settings['filter_stops']=='stops'
    ng_stops = settings['filter_stops']=='ngstops'
    stops = get_stops(settings)

    wd_fds, bg_fds, tg_fds = [], [], []

    for d in processed_collection:
        words = apply_filters(d,settings)
        bgfinder = BigramCollocationFinder.from_words(words, window_size=window)
        tgfinder = TrigramCollocationFinder.from_words(words,window_size=tgwindow)

        if no_stops: #filter stops from words, bigrams and trigrams
            wd_fd = FreqDist({k:v for k, v in bgfinder.word_fd.items() if k not in stops})
            filter_stops(bgfinder,stops)
            filter_stops(tgfinder,stops)
        elif ng_stops: #filter stops from words and bigrams but leave stops in place when mid-word of trigram
            wd_fd = FreqDist({k:v for k, v in bgfinder.word_fd.items() if k not in stops})
            filter_stops(bgfinder,stops)
            filter_tg_stops(tgfinder,stops)
        else:
            wd_fd = bgfinder.word_fd

        if word:
            get_bgcollocates_of_word(bgfinder,word)
            get_tgcollocates_of_word(tgfinder,word)

        wd_fds.append(wd_fd)
        bg_fds.append(bgfinder.ngram_fd)
        tg_fds.append(tgfinder.ngram_fd)

    wd_fd = combine_fds(wd_fds)
    bg_fd = combine_fds(bg_fds)
    tg_fd = combine_fds(tg_fds)

    np_fd = count_features(processed_collection,settings,'np')
    ne_fd = count_features(processed_collection,settings,'ne')
    #custom_fd = count_features(processed_collection,settings,'custom')
    if word:
        return [word,wd_fd[word]],wd_fd,bg_fd,tg_fd
    else:
        return wd_fd,bg_fd,tg_fd,np_fd,ne_fd

def get_top_features(fds,settings):
    nkey = settings['nkey']
    if len(fds) == 4:
        wd_count,wd_fd,bg_fd,tg_fd = fds
        return {
            "top_words": [wd_count],
            "top_bigrams": findtopbigrams(wd_fd,bg_fd,settings),
            "top_trigrams": findtoptrigrams(wd_fd,bg_fd,tg_fd,settings)
        }
    else:
        wd_fd,bg_fd,tg_fd,np_fd,ne_fd = fds
        return {
            "top_words": [[w, wd_fd[w]] for w in sorted(wd_fd, key=wd_fd.get, reverse=True)][:nkey],
            "top_bigrams": findtopbigrams(wd_fd,bg_fd,settings),
            "top_trigrams": findtoptrigrams(wd_fd,bg_fd,tg_fd,settings),
            "top_noun_phrases":  [[np, np_fd[np]] for np in sorted(np_fd, key=np_fd.get, reverse=True)][:nkey],
            "top_named_entities":  [[ne, ne_fd[ne]] for ne in sorted(ne_fd, key=ne_fd.get, reverse=True)][:nkey]
        }

def get_fd(featlist,settings):
    if settings['lcase']=="lcase" or settings['lcase']=="lemma":
        fd = itertoolz.frequencies(feat.lower_ for feat in featlist)
    else:
        fd = itertoolz.frequencies(feat.text for feat in featlist)
    return fd

def get_processed_stats(processed_docs):
    statistics = [get_statistics(doc) for doc in processed_docs]
    return statistics

def get_processed_indices(processed_docs,settings):
    indices = [get_indices(doc,settings) for doc in processed_docs]
    return indices

def get_processed_docs(processed_collection):
    return [d['doc'] for d in processed_collection]

def count_syllables(words, settings):
    specials = [sp.lower() for sp in settings['specialstops']]
    hyphenator = pyphen.Pyphen(lang='en')
    syllables_per_word = [len(hyphenator.positions(word)) + 1 for word in words if word.upper() not in specials]
    n_syllables = sum(syllables_per_word)
    n_polysyllable_words = sum(1 for n in syllables_per_word if n >= 3)
    return n_syllables, n_polysyllable_words

# all the indices
def ttr(n_words,n_unique_words,readability_threshold):
    return 0 if n_words == 0 or n_words < readability_threshold else n_unique_words/n_words

def MTLD(words):
    try:
        return ldiv.mtld(words)
    except ValueError:
        return 0

def hdd(words):
    try:
        return ldiv.hdd(words)
    except ValueError:
        return 0

def lexical_density(n_words,n_content_words,readability_threshold):
    return 0 if n_words == 0 or n_words < readability_threshold else n_content_words/n_words

def smog_measure(n_words,n_polysyllable_words,n_sents,readability_threshold):
    return 0 if n_sents == 0 or n_words < readability_threshold else (1.0430 * sqrt(30 * n_polysyllable_words / n_sents)) + 3.1291

def gunning_fog(n_words,n_sents,n_polysyllable_words,readability_threshold):
    return 0 if n_sents == 0 or n_words == 0 or n_words < readability_threshold else 0.4 * ((n_words / n_sents) + (100 * n_polysyllable_words / n_words))

def flesch_ease(n_words,n_syllables,n_sents,readability_threshold):
    return 0 if n_sents == 0 or n_words == 0 or n_words < readability_threshold else (-84.6 * n_syllables / n_words) - (1.015 * n_words / n_sents) + 206.835

def flesch_kincaid(n_words,n_syllables,n_sents,readability_threshold):
    return 0 if n_sents == 0 or n_words == 0 or n_words < readability_threshold else (11.8 * n_syllables / n_words) + (0.39 * n_words / n_sents) - 15.59

@app.route('/login')
def login():
    if current_user.is_anonymous:
        return render_template('login.html',
                           title = 'Quantext'
                           )
    else:
        return redirect(url_for('control_panel'))

@app.route('/complete_signup', methods=['POST'])
def complete_signup():
    username = request.form.get("username")
    password = request.form.get("password")
    display_name = request.form.get("display_name")
    email = request.form.get("email")
    google_id = request.form.get("google_id")

    if current_user.is_anonymous:
        ph = PasswordHasher()
        hash = ph.hash(password)
        config = {"live_help":"True","control_help":"True","console_help":"True","labeller_help":"True"}
        if google_id != "":
            try:
                user = User.objects.get(google_id=google_id)
                user['username'] = username
                user['password'] = hash
                user['display_name'] = display_name
                user['email'] = email
                user['google_id'] = google_id
                user.save()
            except:
                user = User(username=username,password=hash,display_name=display_name,email=email,google_id=google_id,config=config).save()
        else:
            user = User(username=username,password=hash,display_name=display_name,email=email,google_id=google_id,config=config).save()

        login_user(user, True)

    return redirect(url_for('control_panel'))

@app.route('/canvas_oauth', methods=['POST'])
def canvas_oauth():
    key = request.form.get('oauth_consumer_key')
    canvas_id = request.form.get('custom_canvas_user_login_id')
    display_name = request.form.get('lis_person_name_full')
    email = request.form.get('lis_person_contact_email_primary')

    if current_user.is_anonymous:
        try:
            user = User.objects.get(canvas_id=canvas_id)
        except:
            config = {"live_help":"True","control_help":"True","console_help":"True","labeller_help":"True"}
            user = User(display_name=display_name,email=email,canvas_id=canvas_id,config=config).save()

        login_user(user, True)
        return redirect(url_for('login'))
    else:
        if canvas_id:
            current_user['canvas_id'] = canvas_id
        current_user.save()
        return redirect(url_for('control_panel'))

@app.route('/log_me_in', methods=['POST'])
def log_me_in():
    username = request.form.get("username")
    password = request.form.get("password")
    captcha = request.form.get("captcha")
    res = requests.post("https://www.google.com/recaptcha/api/siteverify",{"secret":"6Lf4OJoUAAAAAFgLkK5P5zRGo-sF9jjJj4saIbPm","response":captcha})
    response_data = res.json()
    if response_data["success"]:
        try:
            user = User.objects.get(username=username)
        except:
            print("No user found")
            return render_template('login.html',
                                   title = 'Quantext',
                                   error = "Oops... incorrect username or password."
                                   )
        if username != "lokeshpadhye" and username != "cwhittaker" and username != "AndrewWithy":
            ph = PasswordHasher()
            try:
                ph.verify(user["password"], password)
                login_user(user, True)
                return redirect(url_for('control_panel'))
            except VerifyMismatchError:
                print("PASSWORD ERROR")
                return render_template('login.html',
                                       title = 'Quantext',
                                       error = "Oops... incorrect username or password."
                                       )
        else:
            if password == user["password"]:
                login_user(user, True)
                return redirect(url_for('control_panel'))
    else:
        print("Captcha error")
        return render_template('login.html',
                               title = 'Quantext',
                               error = "Oops... something went wrong with the reCAPTCHA. Please contact the site administrator."
                               )

@app.route('/check_username/<username>', methods=["GET"])
def check_username(username=None):
    if username:
        try:
            user = User.objects.get(username=username)
            return "User exists"
        except:
            return "User not found"

@app.route('/check_email/<email>', methods=["GET"])
def check_email(email=None):
    if email:
        try:
            user = User.objects.get(email=email)
            return "User exists"
        except:
            return "User not found"

@app.route('/blacklist_question', methods=['POST'])
def blacklist_question():
    analysis_id = request.get_json()["analysis_id"]
    file_id = request.get_json()["file_id"]
    question_number = request.get_json()["question_number"]
    black_or_white = request.get_json()["black_or_white"]
    if not analysis_id is None and not file_id is None and not question_number is None:
        file = File.objects.get(id=file_id)
        cuid = str(file['owner']['id'])
        analysis = Analysis.objects.get(id=analysis_id)
        blist = analysis['files'][file_id]["columns"][question_number]['settings'][black_or_white]
        
        idx = int(question_number)
        custom = file["columns"][idx]
        d = nlp(custom)
        words_doc = get_words(d)
        words = [word.lower_ for word in words_doc]

        for word in words:
            if word not in blist:
                blist.append(word)

        analysis['files'][file_id]["columns"][question_number]['settings'][black_or_white] = blist
        analysis.save()

        top_features, question = reprocessFeatures(cuid,analysis_id,file_id,question_number)            
        keywords = render_template('keywords.html', analysis=analysis,file=file,question=question,features=top_features,current_user=current_user)
        question_settings = render_template('question_settings.html',analysis=analysis,file=file,question=question)
        return json.dumps({"keywords":keywords,"question_settings":question_settings})

#admin function
@app.route('/update_user', methods=['POST'])
def update_user():
    email = request.get_json()["email"]
    plan = request.get_json()["plan"]
    user = User.objects.get(email=email)
    user['plan'] = plan
    user.save()
    return "Success"

#user function
@app.route('/update_profile', methods=['POST'])
def update_profile():
    email = request.get_json()["email"]
    display_name = request.get_json()["display_name"]
    if not current_user.is_anonymous:
        current_user['email'] = email
        current_user['display_name'] = display_name
        current_user.save()
    return "Success"

@app.route('/', methods=['GET'])
def control_panel():
    activity = defaultdict()
    if not current_user.is_anonymous:
        lastrun_analysis = Analysis.objects(owner=current_user['id']).order_by('-lastrun')
        if lastrun_analysis:
            lastrun_analysis = lastrun_analysis[0]

        all_analyses = Analysis.objects(owner=current_user['id']).order_by('-created')
        currYear = "2020"

        maxNum = 0
        for a in all_analyses: 
            year = a['created'].strftime('%Y')
            month = int(a['created'].strftime('%m')) - 1
            day = int(a['created'].strftime('%d')) - 1

            month = str(month)
            day = str(day)

            if year == currYear:
                if not year in activity:
                    activity[year] = defaultdict()
                
                if not month in activity[year]:
                    activity[year][month] = defaultdict()

                if not day in activity[year][month]:
                    activity[year][month][day] = 0
                
                activity[year][month][day] += 1
                if activity[year][month][day] > maxNum:
                    maxNum = activity[year][month][day]

        quantext_files = File.objects(owner=current_user['id'], status__ne='deleted', file_type="quantext").order_by('-created')
        live_files = File.objects(owner=current_user['id'], status__ne='deleted', file_type="quantext_live").order_by('-created')

        return render_template('controlPanel.html',
                               title = 'Quantext',
                               analyses = all_analyses,
                               activity=activity,
                               maxNum=maxNum,
                               quantext_files = quantext_files,
                               live_files = live_files,
                               lastrun_analysis=lastrun_analysis)
    else:
        return redirect(url_for('login'))

@app.route('/livePanel')
def live_panel():
    if not current_user.is_anonymous:
        analyses = Analysis.objects(owner=current_user['id'], status__ne='deleted', type="live").order_by('-created')
        all_columns = []
        for l in analyses:
            for k,v in l["files"].items():
                f = {"file_id":k}
                pickle_path = os.path.join(app.config['PANDAS_FOLDER'],k,str(l['id'])+"_data.pkl")
                try:
                    u = URL.objects.get(analysis_id=str(l['id']),file_id=k)
                    key = u.key
                    df = nlputility.read_pickle(pickle_path)
                    f["list"] = df.to_dict(orient='list')
                    all_columns.append([df.columns[0],key])
                except:
                    pass

        return render_template('livePanel.html',
                               current_user=current_user,
                               all_columns=all_columns,
                               live=analyses,
                               title = 'Quantext Live')
    else:
        return redirect(url_for('login'))

@app.route('/live_ajax')
def live_ajax():
    if not current_user.is_anonymous:
        live = File.objects(owner=current_user['id'], status__ne='deleted', file_type="quantext_live").order_by('-created')
        participants = []
        for l in live:
            l_key = "live_" + str(l['id'])
            if r.exists(l_key):
                participants.append(r.get(l_key).decode("utf-8"))
            else:
                participants.append("False")

        return json.dumps(participants)

@app.route('/adminPanel')
def admin_panel():
    if not current_user.is_anonymous and current_user['isAdmin'] == "true":
        users = User.objects
        user_list = []
        for u in users:
            user = {"user":u}
            all_analyses = Analysis.objects(owner=u['id']).order_by('-lastrun')
            user['analyses'] = all_analyses
            user_list.append(user)
        return render_template('adminPanel.html',
                           title = 'Quantext',
                           users = user_list
                           )
    else:
        return render_template('404.html'), 404

@app.route('/logout')
def logout():
    logout_user()
    return redirect("https://quantext.org")

@app.route('/authorize', methods = ['POST'])
def authorize():
    token = request.form.get("idtoken")
    try:
        # Specify the CLIENT_ID of the app that accesses the backend:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), app.config['OAUTH_CREDENTIALS']['google']['web']['client_id'])
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')

        google_id = idinfo['sub']
        email = idinfo['email']
        name = idinfo['name']

        try:
            user = User.objects.get(google_id=google_id)
        except:
            config = {"live_help":"True","control_help":"True","console_help":"True","labeller_help":"True"}
            user = User(google_id=google_id, display_name=name, email=email, config=config).save()

        login_user(user, True)
        return redirect(url_for('control_panel'))
    except ValueError:
        # Invalid token
        print("???")
        return redirect("https://quantext.org")

@app.route('/link_account', methods = ['POST'])
def link_account():
    token = request.form.get("idtoken")
    try:
        # Specify the CLIENT_ID of the app that accesses the backend:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), app.config['OAUTH_CREDENTIALS']['google']['web']['client_id'])
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')

        google_id = idinfo['sub']
        return google_id
    except ValueError:
        # Invalid token
        print("???")
        return "ERROR"

@app.route('/authorize/<provider>')
def oauth_authorize(provider):
    if not current_user.is_anonymous:
        return redirect(url_for('control_panel'))
    if provider == 'twitter':
        oauth = OAuthSignIn.get_provider(provider)
        return oauth.authorize()
    else:
        return redirect(url_for('google_callback'))

@app.route('/callback/<provider>')
def oauth_callback(provider):
    if not current_user.is_anonymous:
        return redirect(url_for('control_panel'))

    oauth = OAuthSignIn.get_provider(provider)
    twitter_id, username, email = oauth.callback()
    if twitter_id is None:
        flash('Authentication failed.')
        return redirect("https://quantext.org")

    try:
        user = User.objects.get(twitter_id=twitter_id)
    except:
        config = {"live_help":"True","control_help":"True","console_help":"True","labeller_help":"True"}
        user = User(twitter_id=twitter_id, display_name=username, email=email, config=config).save()

    login_user(user, True)
    return redirect(url_for('control_panel'))

@celery.task(bind=True)
def upload_file(self,file_id,filename,folder):
    self.update_state(state='PROGRESS',meta={'current_task':'Uploading file','process_percent':1})

    file = File.objects.get(id=file_id)
    origpath = os.path.join(app.config['CORPUS_FOLDER'],file_id)
    if not os.path.exists(origpath):
        os.makedirs(origpath)
    df = nlputility.getdata(filename,folder)
    file["columns"] = list(df.columns)
    file["rows"] = len(df.index)

    self.update_state(state='PROGRESS',meta={'current_task':'Uploading file','process_percent':35})

    pickle_path = os.path.join(app.config['PANDAS_FOLDER'],file_id)
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)

    self.update_state(state='PROGRESS',meta={'current_task':'Pickling','process_percent':65})

    df.to_pickle(os.path.join(pickle_path,'data.pkl'))
    self.update_state(state='PROGRESS',meta={'current_task':'Finishing','process_percent':100})
    file.save()
    return "Success"


@celery.task(bind=True)
def process_reference(self,file_id,filename,folder):
    self.update_state(state='PROGRESS',meta={'process_percent':10})
    p = os.path.join(app.config[folder],filename)
    text = pdf_to_text(p)
    file = File.objects.get(id=file_id)
    file['text'] = text
    self.update_state(state='PROGRESS',meta={'process_percent':40})

    reference_statistics = process_reference_statistics(text)

    self.update_state(state='PROGRESS',meta={'process_percent':80})
    file['statistics'] = reference_statistics[0]
    self.update_state(state='PROGRESS',meta={'process_percent':100})
    file.save()

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

#file upload
@app.route('/poll_state', methods =['POST'])
def poll_state():
    task_id = request.get_json()['task_id']
    task = upload_file.AsyncResult(task_id)
    data = task.result
    return json.dumps(data)

@app.route('/poll_analysis', methods =['POST'])
def poll_analysis():
    task_id = request.get_json()['task_id']
    task = process_one.AsyncResult(task_id)
    data = task.result
    return json.dumps(data)

@app.route('/uploadq', methods = ['POST'])
def upload_file_qs():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            try:
                file_id, job = None, None
                filename = fname_generator(secure_filename(f.filename))
                path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
                f.save(path)

                if filename.rsplit('.', 1)[1] == "csv":
                    with open(path, 'rb') as utffile:
                        content_bytes = utffile.read()
                    detected = chardet.detect(content_bytes)
                    encoding = detected['encoding']                    
                    content_text = content_bytes.decode(encoding)
                    with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(path),
                                                    encoding='utf-8', delete=False) as utffile:
                        utffile.write(content_text)
                    os.replace(utffile.name, path)

                user = User.objects.get(id=current_user.id)

                file = File(filename=filename,owner=user,status="processing")
                if allowed_fileq(f.filename):
                    filesize = os.path.getsize(path)
                    file['filesize'] = int(filesize)
                    file['file_type'] = 'quantext'
                    file.save()
                    file_id = file['id']
                    job = upload_file.delay(str(file_id),filename,'UPLOAD_FOLDER')
                elif allowed_filet(f.filename):
                    file['file_type'] = 'reference'
                    file.save()
                    file_id = file['id']
                    job = process_reference.delay(str(file_id),filename,'UPLOAD_FOLDER')

                json_data = {"job_id":job.id,"f":{"filename":file["filename"],"id":str(file_id)}}
                return json.dumps(json_data), 200
            except Exception as e:
                return 'There is a problem uploading your file: ' + str(e), 500

        else:
            return 'Your file must have a .xls or .xlsx extension', 500

    return 'Failed', 500

@app.route('/file_data/<file_id>', methods = ['GET'])
def file_data(file_id=None):
    file = File.objects.get(id = file_id)
    origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id)    
    file_path = os.path.join(origpath,"data.pkl")
    df = nlputility.read_pickle(file_path)
    df = df.head(5).fillna("")

    u = URL.objects(file_id=file_id)
    urls = []

    for x in u:
        if x['analysis_id'] not in urls:
            urls.append(x['analysis_id'])

    return render_template('file_data.html',urls=urls, file=file, spreadsheet=df.to_dict('split'))

@app.route('/add_file_data/<file_id>', methods = ['GET'])
def add_file_data(file_id=None):
    file = File.objects.get(id = file_id)
    origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id)    
    file_path = os.path.join(origpath,"data.pkl")
    df = nlputility.read_pickle(file_path)
    df = df.head(5).fillna("")

    return render_template('partials/add_files.html', file=file, spreadsheet=df.to_dict('split'))

@app.route('/analysis_data/<analysis_id>', methods = ['GET'])
def analysis_data(analysis_id=None):
    analysis = Analysis.objects.get(id = analysis_id)
    return render_template('partials/analysis_box.html', analysis=analysis)

@app.route('/uploads/<filename>')
def uploaded_files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/newAnalysis', methods = ['POST'])
def new_analysis():
    files = request.get_json()['files']
    name = request.get_json()['name']
    list_of_users = request.get_json()['list_of_users']
    list_of_users = list_of_users.split(",")
    list_of_users = [x for x in list_of_users if x != ""]

    user = User.objects.get(id=current_user.id)
    analysis = Analysis(owner=user, type="normal")
    analysis['name'] = name
    analysis['to_share'] = list_of_users

    already_in_system = User.objects(email__in=list_of_users)
    if len(already_in_system) > 0:
        analysis["shared"] = already_in_system

    if files:
        analysis["files"] = files

    analysis.save()
    return str(analysis.id)

@app.route("/save_live_response", methods=["POST"])
def save_live_response():
    response = request.get_json()['response']
    analysis_id = request.get_json()['analysis_id']
    file_id = request.get_json()['file_id']
    question_number = request.get_json()['question_number']
    if not analysis_id is None and not file_id is None and not question_number is None:
        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        cuid = str(file['owner']['id'])
        idx = int(question_number)

        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+"_data.pkl")
        df = nlputility.read_pickle(pickle_path)
        col = df.columns[idx]
        df = df.append({col:response}, ignore_index=True)
        df.to_pickle(os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+"_data.pkl"))

        question = {"number":question_number, "text":col}

        r_doc = nlp(response)
        origpath = os.path.join(app.config['DOC_FOLDER'],file_id,question_number)
        if not os.path.exists(origpath):
            os.makedirs(origpath)
        origpath = os.path.join(app.config['DOC_FOLDER'],file_id,question_number,str(len(df[df.columns[idx]].dropna().tolist())-1))
        r_doc.doc.to_disk(origpath)

        r_key = cuid + "_" + file_id
        nlp_obj = pickle.loads(r.get(r_key))
        docs = load_docs_from_disk(file_id,question_number,len(df[df.columns[idx]].dropna().tolist()))
        processed_settings = process_settings(analysis['files'][file_id]["columns"][question_number]['settings'])
        top_features = get_top_features(get_features(docs,processed_settings,None),processed_settings)
        nlp_obj["top_features"] = top_features
        r.set(r_key,pickle.dumps(nlp_obj))

        return render_template('index_keywords.html',
                               textlist=df[df.columns[idx]].dropna().tolist(),
                               analysis=analysis,
                               file=file,
                               question=question,
                               top_features=top_features
                           )

@app.route("/poll_live_responses/<key>/<type>", methods=["GET"])
def poll_live_responses(key=None,type=None):
    if not key is None:
        u = URL.objects.get(key=key)
        analysis_id = u.analysis_id
        file_id = u.file_id
        question_number = u.question_number

        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        cuid = str(file['owner']['id'])
        idx = int(question_number)

        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_data.pkl")
        df = nlputility.read_pickle(pickle_path)
        col = df.columns[idx]

        r_key = cuid + "_" + file_id
        nlp_obj = pickle.loads(r.get(r_key))
        top_features = nlp_obj["top_features"]

        question = {"number":question_number, "text":col}
        responses = df[col].dropna().tolist()

        top = render_template('index_keywords_top_features.html',
                        responses=responses,
                        analysis=analysis,
                        file=file,
                        question=question,
                        top_features=top_features)

        right = render_template('all_responses_admin.html',
                              responses=responses,
                              analysis=analysis,
                              file=file,
                                key=key,
                              question=question,
                              top_features=top_features)

        return json.dumps({"top":top,"right":right})

@app.route("/live_top_data/<key>/<type>", methods=["GET"])
def live_top_data(key=None,type=None):
    if not key is None:
        u = URL.objects.get(key=key)
        analysis_id = u.analysis_id
        file_id = u.file_id
        question_number = u.question_number

        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        cuid = str(file['owner']['id'])
        idx = int(question_number)

        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_data.pkl")
        df = nlputility.read_pickle(pickle_path)
        col = df.columns[idx]

        r_key = cuid + "_" + file_id
        nlp_obj = pickle.loads(r.get(r_key))
        top_features = nlp_obj["top_features"]

        l_key = "live_" + file_id
        question = {"number":question_number, "text":col}
        responses = df[col].dropna().tolist()
        if type == "user":
            r.set(l_key,"True")
            r.expire(l_key,60)
            view = "index_keywords.html"
        else:
            view = "index_keywords_admin.html"

        return render_template(view,
                              responses=responses,
                              analysis=analysis,
                              file=file,
                              question=question,
                              key=key,
                              top_features=top_features)

@app.route("/new_quantext_live", methods=["POST"])
def new_quantext_live():
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        user = User.objects.get(id=current_user.id)

        text = request.get_json()['text']
        file = File(filename=text,owner=user,status="processing",file_type="quantext_live")
        file["columns"] = [text]
        file.save()
        file_id = str(file['id'])

        analysis = Analysis(owner=user, type="live")
        analysis['name'] = text + " (Quantext Live)"
        analysis.lastrun = datetime.datetime.now
        analysis['files'] = {}
        analysis['files'][file_id] = {"columns":{"0":{}},"filename":text}

        df = nlputility.new_dataframe()
        df[text] = []

        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],file_id)
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        df.to_pickle(os.path.join(pickle_path,'data.pkl'))

        s = default_settings()
        show_freq = request.get_json()['show_freq']
        public = request.get_json()['public']
        s['show_freq'] = show_freq
        s['public'] = public

        analysis['files'][file_id]["columns"]["0"]["settings"] = s
        analysis['files'][file_id]["columns"]["0"]["categories"] = {"all":{},"Categories_1":{}}
        analysis['files'][file_id]["columns"]["0"]["label_settings"] = {"Categories_1":{"randomise_labels":False,"hide_labels":False}}
        analysis.save()

        origpath = os.path.join(app.config['DOC_FOLDER'],file_id)
        if not os.path.exists(origpath):
            os.makedirs(origpath)

        origpath = os.path.join(app.config['DOC_FOLDER'],file_id,"0")
        if not os.path.exists(origpath):
            os.makedirs(origpath)

        docs = []
        for r_idx,response in enumerate(df[df.columns[0]].dropna().tolist()):
            response = process_response(str(response))
            docs.append(process_doc(file_id,"0",str(r_idx),response))

        processed_settings = process_settings(s)
        top_features = {"0":get_top_features(get_features(docs,processed_settings,None),processed_settings)}
        stats = get_processed_stats(docs)
        pos = [get_pos(get_words(doc)) for doc in docs]
        cc = df.columns[0]

        col_df = nlputility.new_dataframe()
        col_df['Categories_1'] = [[]]*len(df)
        col_df['Characters'] = [np.nan]*len(df)
        col_df['Words'] = [np.nan]*len(df)
        col_df['Sentences'] = [np.nan]*len(df)
        col_df['POS'] = [""]*len(df)

        col_df['TTR'] = [np.nan]*len(df)
        col_df['MTLD'] = [np.nan]*len(df)
        col_df['HDD'] = [np.nan]*len(df)
        col_df['LD'] = [np.nan]*len(df)
        col_df['SMOG'] = [np.nan]*len(df)
        col_df['Gunning'] = [np.nan]*len(df)
        col_df['Flesch'] = [np.nan]*len(df)
        col_df['FK'] = [np.nan]*len(df)

        count = 0
        for index, row in df[cc].dropna().iteritems():
            col_df.at[index,'Characters'] = stats[count]['Characters']
            col_df.at[index,'Words'] = stats[count]['Words']
            col_df.at[index,'Sentences'] = stats[count]['Sentences']
            col_df.at[index,'POS'] = pos[count]
            count+=1

        analysis_id = str(analysis['id'])
        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],file_id)
        df.to_pickle(os.path.join(pickle_path,analysis_id+'_data.pkl'))
        col_df.to_pickle(os.path.join(pickle_path,analysis_id+'_0.pkl'))

        key = get_time_stamp_URL("0")
        URL(analysis_id=analysis_id,file_id=file_id,question_number="0",key=key).save()

        nlp_obj = {"top_features":top_features}
        r_key = cuid + "_" + file_id
        r.set(r_key,pickle.dumps(nlp_obj))
        analysis.save()

        return render_template('quantext_live.html', l=file)

@app.route('/get_deleted_files/<analysis_id>', methods =['GET'])
def get_deleted_files(analysis_id=None):
    if not analysis_id is None:
        deleted = False
        analysis = Analysis.objects.get(id=analysis_id)
        for f in analysis['files'].keys():
            file = File.objects.get(id=f)
            if file['status'] == 'deleted':
                deleted = True

        if deleted:
            return "deleted"
        else:
            return "not deleted"

@app.route('/get_quantext_files/<type>', methods =['GET'])
def get_quantext_files(type=None):
    if not current_user.is_anonymous:
        files = File.objects(owner=current_user['id'], status__ne='deleted', file_type=type).order_by('-created')
        if len(files) > 0:
            return render_template("all_student_files.html", files=files)
        else:
            return render_template("partials/empty.html", list="files")

@app.route('/deleteAnalysis', methods = ['POST'])
def delete_analysis():
    analysis_id = request.get_json()['id']
    an = Analysis.objects.get(id=analysis_id)
    an.delete()
    return render_template('partials/no_details.html')

@app.route('/deleteFile', methods = ['POST'])
def delete_file():
    file_id = request.get_json()['id']
    file = File.objects.get(id=file_id)
    file['status'] = "deleted"
    file.save()
    return render_template('partials/no_details.html')

@app.route("/save_theme", methods = ['POST'])
def save_theme():
    theme = request.get_json()["theme"]
    if not current_user.is_anonymous:
        current_user.theme = theme
        current_user.save()
    return "Success"

@app.route('/analyse/', methods = ['GET'])
@app.route('/analyse/<analysis_id>', methods = ['GET'])
def analyse(analysis_id=None):
    if not current_user.is_anonymous:
        if not analysis_id is None:
            analysis = Analysis.objects.get(id=analysis_id)
            file_id = list(analysis['files'])[0]
            return render_template("analysis_loading.html",analysis=analysis,file_id=file_id)
    else:
        return redirect(url_for('control_panel'))

@app.route('/analysis/<key>', methods = ['GET'])
def analysis(key=None):
    if not current_user.is_anonymous:
        if not key is None:
            u = URL.objects.get(key=key)
            analysis_id = u.analysis_id
            file_id = u.file_id
            analysis = Analysis.objects.get(id=analysis_id)
            remove_to_share = [u["email"] for u in analysis["shared"]]
            analysis["to_share"] = [u for u in analysis["to_share"] if u not in remove_to_share]

            if not file_id is None:
                file = File.objects.get(id=file_id)
                columns_to_analyse = analysis['files'][file_id]["columns"]
                character_total = 0
                word_total = 0
                sentence_total = 0
                new_columns = []
                for i in sorted(columns_to_analyse.keys()):
                    colpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_'+str(i)+'.pkl')
                    col_df = nlputility.read_pickle(colpath)
                    character_total += sum(col_df['Characters'].dropna().tolist())
                    word_total += sum(col_df['Words'].dropna().tolist())
                    sentence_total += sum(col_df['Sentences'].dropna().tolist())
                    new_columns.append({"text":file["columns"][int(i)],"number":i})

                totals = [character_total,word_total,sentence_total]

                return flask_response(stream_with_context(stream_template('analyse.html',analysis=analysis,file=file,statistics=totals,columns=new_columns)))

def createCharts(statistics,keys,decimals):
    h = {key : createChart(statistics,decimals[idx]) for idx,key in enumerate(keys)}
    return h

def createChart(statistics,decimal):
    _bins = 20
    _list = [float(x) for x in statistics]
    if not _list is None and len(_list) > 0:
    #_min = np.min(_list)
        _min = 0
        _max = np.max(_list)
        _diff = 0 if isnan(_max) or isnan(_min) else _max - _min

        if _diff <= _bins:
            _step = round(1/_bins,2) if decimal else 1
        else:
            _step = ceil(_diff/_bins)

        bins = [i*_step for i in range(0,_bins+1)]
        _hist = np.histogram(_list, bins=bins)
        mx = np.max(_hist[0]) if _hist[0].any() else 0
        return [_hist,mx,_step]
    else:
        return []

alpha = "_GcEeghijkNlBnWXqdstbLIxywzAfCDuFrHZJKpMOPaQURTSmvVoY"

def get_time_stamp_URL(question_number):
    ts = time.time()
    ts = floor(ts)
    ts = str(ts)

    final = ""
    pos_to_skip = None

    for idx,c in enumerate(ts):
        if pos_to_skip is None:
            try:
                pos_to_skip = idx+1
                next = ts[pos_to_skip]
                combined = c+next

                if int(combined) < 52:
                    final += alpha[int(combined)]
                else:
                    final += alpha[int(c)]
                    pos_to_skip = None
            except:
                final += alpha[int(c)]
                pos_to_skip = None
        else:
            pos_to_skip = None

    final = final + question_number
    return final

@celery.task(bind=True)
def process_one(self,cuid,analysis_id):
    curr_percent = 1
    self.update_state(state='PROGRESS',meta={'current_task':'Initialising session','process_percent':curr_percent})
    return_key = ""
    if not analysis_id is None:
        analysis = Analysis.objects.get(id=analysis_id)
        analysis.lastrun = datetime.datetime.now
        analysis.save()
        curr_percent = 10
        outer_step = 90/len(analysis["files"])
        for file_id,file in analysis["files"].items():
            if not file_id is None:
                file_object = File.objects.get(id=file_id)
                file["filename"] = file_object['filename']
                r_key = cuid + "_" + file_id
                origpath = os.path.join(app.config['DOC_FOLDER'],file_id)
                if not os.path.exists(origpath):
                    os.makedirs(origpath)
                self.update_state(state='PROGRESS',meta={'current_task':'Loading text','process_percent':curr_percent})
                origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id)

                load_path = os.path.join(origpath,analysis_id+"_data.pkl")
                new_path = os.path.join(origpath,"data.pkl")
                pickle_path = os.path.join(app.config['PANDAS_FOLDER'],file_id)

                try:
                    df = nlputility.read_pickle(load_path)
                    u = URL.objects(analysis_id=analysis_id,file_id=file_id)[0]
                    return_key = u["key"]
                    
                except:
                    df = nlputility.read_pickle(new_path)
                    columns_to_analyse = file["columns"]

                    nlp_obj = {"top_features":{}}
                    total_questions = len(columns_to_analyse)
                    step = outer_step/total_questions/4
                    top_features = {}

                    file["columns"] = {}
                    for idx,col in enumerate(columns_to_analyse):
                        key = get_time_stamp_URL(col)
                        return_key = key
                        URL(key=key,analysis_id=analysis_id,file_id=file_id,question_number=col).save()

                        col_df = nlputility.new_dataframe()
                        s = default_settings()

                        file["columns"][col] = {}
                        file["columns"][col]["settings"] = s
                        file["columns"][col]["categories"] = {"all":{},"Categories_1":{}}
                        file["columns"][col]["label_settings"] = {"Categories_1":{"randomise_labels":False,"hide_labels":False}}

                        curr_percent += step
                        self.update_state(state='PROGRESS',meta={'current_task':'Processing text ('+str(idx+1) + '/'+str(total_questions)+')','process_percent':curr_percent})

                        origpath = os.path.join(app.config['DOC_FOLDER'],file_id,col)
                        if not os.path.exists(origpath):
                            os.makedirs(origpath)

                        docs = []
                        for r_idx,response in enumerate(df[df.columns[int(col)]].dropna().tolist()):
                            response = process_response(str(response))
                            docs.append(process_doc(file_id,col,str(r_idx),response))

                        processed_settings = process_settings(s)
                        curr_percent += step
                        self.update_state(state='PROGRESS',meta={'current_task':'Generating top features ('+str(idx+1) + '/'+str(total_questions)+')','process_percent':curr_percent})
                        top_features[col] = get_top_features(get_features(docs,processed_settings,None),processed_settings)
                        curr_percent += step
                        self.update_state(state='PROGRESS',meta={'current_task':'Generating statistics ('+str(idx+1) + '/'+str(total_questions)+')','process_percent':curr_percent})

                        stats = get_processed_stats(docs)
                        indices = get_processed_indices(docs,processed_settings)
                        pos = [get_pos(get_words(doc)) for doc in docs]
                        cc = df.columns[int(col)]

                        col_df['Categories_1'] = [[]]*len(df)
                        col_df['Characters'] = [np.nan]*len(df)
                        col_df['Words'] = [np.nan]*len(df)
                        col_df['Sentences'] = [np.nan]*len(df)
                        col_df['POS'] = [""]*len(df)

                        col_df['TTR'] = [np.nan]*len(df)
                        col_df['MTLD'] = [np.nan]*len(df)
                        col_df['HDD'] = [np.nan]*len(df)
                        col_df['LD'] = [np.nan]*len(df)
                        col_df['SMOG'] = [np.nan]*len(df)
                        col_df['Gunning'] = [np.nan]*len(df)
                        col_df['Flesch'] = [np.nan]*len(df)
                        col_df['FK'] = [np.nan]*len(df)

                        count = 0
                        self.update_state(state='PROGRESS',meta={'current_task':'Generating indices ('+str(idx+1) + '/'+str(total_questions)+')','process_percent':curr_percent})
                        for index, row in df[cc].dropna().iteritems():
                            col_df.at[index,'Characters'] = stats[count]['Characters']
                            col_df.at[index,'Words'] = stats[count]['Words']
                            col_df.at[index,'Sentences'] = stats[count]['Sentences']
                            col_df.at[index,'POS'] = pos[count]

                            col_df.at[index,'TTR'] = indices[count]["TTR"]
                            col_df.at[index,'MTLD'] = indices[count]["MTLD"]
                            col_df.at[index,'HDD'] = indices[count]["HDD"]
                            col_df.at[index,'LD'] = indices[count]["LD"]
                            col_df.at[index,'SMOG'] = indices[count]["SMOG"]
                            col_df.at[index,'Gunning'] = indices[count]["Gunning"]
                            col_df.at[index,'Flesch'] = indices[count]["Flesch"]
                            col_df.at[index,'FK'] = indices[count]["FK"]

                            count+=1

                        col_df.to_pickle(os.path.join(pickle_path,analysis_id+'_'+col+'.pkl'))

                    df.to_pickle(os.path.join(pickle_path,analysis_id+'_data.pkl'))

                    nlp_obj["top_features"] = top_features
                    curr_percent += step
                    self.update_state(state='PROGRESS',meta={'current_task':'Finalising','process_percent':curr_percent})
                    r.set(r_key,pickle.dumps(nlp_obj))
        analysis.save()
        return return_key


@app.route('/analyse_task/<analysis_id>', methods=['GET'])
def analyse_task(analysis_id=None):
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        job = process_one.delay(cuid,analysis_id)
        json_data = {"job_id":job.id}
        return json.dumps(json_data), 200
    else:
        return redirect(url_for('control_panel'))

@app.route('/analyse/<analysis_id>/<file_id>', methods=['GET'])
def analyse_files(analysis_id=None,file_id=None):
    if not current_user.is_anonymous:
        if not analysis_id is None and not file_id is None:
            u = URL.objects(analysis_id=analysis_id,file_id=file_id)[0]
            key = u.key
            return redirect(url_for('analysis',key=key))

@app.route('/reference_materials/<analysis_id>')
def reference_materials(analysis_id=None):
    if not analysis_id is None:
        analysis = Analysis.objects.get(id=analysis_id)
        return render_template('reference_materials.html', analysis=analysis)

from collections import OrderedDict

@app.route('/label/<key>', methods=['GET'])
def label(key=None):
    if not key is None:
        number = key[-1:]
        key = key[:-1]

        cat_num = alpha.index(number)
        cat_col = "Categories_"+str(cat_num)

        u = URL.objects.get(key=key)
        analysis_id = u.analysis_id
        file_id = u.file_id
        question_number = u.question_number

        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        cuid = str(file['owner']['id'])
        idx = int(question_number)

        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_data.pkl")
        df = nlputility.read_pickle(pickle_path)
        col = df.columns[idx]

        colpath = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_"+question_number+".pkl")
        col_df = nlputility.read_pickle(colpath)

        merge_df = nlputility.merge(df,col_df)
        merge_df = merge_df.filter(items=[col,'Characters','Words','Sentences','POS',cat_col]).dropna(axis=0)
        tuples = list(merge_df.itertuples(index=True, name=None))

        cat_cols = sorted([col for col in col_df.columns if 'Categories_' in col])

        col_df = col_df.filter(items=cat_cols).dropna(axis=0)
        total = len(df[col].dropna().tolist())
        percents, marked_up = [], []
        
        m = len([x for x in col_df[cat_col].dropna().tolist() if len(x) != 0])
        percent = m/total*100
        marked_up.append(m)
        percents.append(percent)

        question = {"number":question_number, "text":col}

        alias = analysis['files'][file_id]["columns"][question_number]["settings"]["aliases"][cat_col]

        arr = analysis['files'][file_id]["columns"][question_number]["categories"]["all"]
        arr2 = [str(x) for x in sorted([int(y) for y in arr])]
        arr3 = {}
        
        for i in arr2:
            arr3[i] = arr[i]

        analysis['files'][file_id]["columns"][question_number]["categories"]["all"] = arr3

        return render_template('label_user.html',
                               alias=alias,
                               key=key,
                               analysis=analysis,
                               file=file,
                               question=question,
                               cat_cols=cat_cols,
                               cat_col=cat_col,
                               marked_up=marked_up,
                               percents=percents,
                               tuples=tuples,
                               cat_num=cat_num
                               )

@app.route('/labeller_manual/<key>', methods=['GET'])
def labeller_manual(key=None):
    if not key is None:
        number = key[-1:]
        key = key[:-1]

        cat_num = alpha.index(number)
        cat_col = "Categories_"+str(cat_num)

        u = URL.objects.get(key=key)
        analysis_id = u.analysis_id
        file_id = u.file_id
        question_number = u.question_number

        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        cuid = str(file['owner']['id'])
        idx = int(question_number)

        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_data.pkl")
        df = nlputility.read_pickle(pickle_path)
        col = df.columns[idx]

        colpath = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_"+question_number+".pkl")
        col_df = nlputility.read_pickle(colpath)

        merge_df = nlputility.merge(df,col_df)
        merge_df = merge_df.filter(items=[col,'Characters','Words','Sentences','POS',cat_col]).dropna(axis=0)
        tuples = list(merge_df.itertuples(index=True, name=None))

        cat_cols = sorted([col for col in col_df.columns if 'Categories_' in col])

        col_df = col_df.filter(items=cat_cols).dropna(axis=0)
        total = len(df[col].dropna().tolist())
        percents, marked_up = [], []

        m = len([x for x in col_df[cat_col].dropna().tolist() if len(x) != 0])
        percent = m/total*100
        marked_up.append(m)
        percents.append(percent)

        question = {"number":question_number, "text":col}

        return render_template('labeller_manual.html',
                               key=key,
                               analysis=analysis,
                               file=file,
                               question=question,
                               cat_cols=cat_cols,
                               cat_col=cat_col,
                               marked_up=marked_up,
                               percents=percents,
                               tuples=tuples
                               )

@app.route('/live/<key>', methods=['GET'])
def live(key=None):
    if not key is None:
        u = URL.objects.get(key=key)
        analysis_id = u.analysis_id
        file_id = u.file_id
        question_number = u.question_number

        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        cuid = str(file['owner']['id'])
        idx = int(question_number)

        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_data.pkl")
        df = nlputility.read_pickle(pickle_path)
        col = df.columns[idx]

        r_key = cuid + "_" + file_id
        nlp_obj = pickle.loads(r.get(r_key))
        top_features = nlp_obj["top_features"]

        question = {"number":question_number, "text":col}

        return render_template('live.html',
                               top_features=top_features,
                               analysis=analysis,
                               file=file,
                               key=key,
                               question=question
                               )

@app.route('/qrcode/<key>', methods=['GET'])
def qrcode(key=None):
    if not key is None:
        return render_template('qrcode.html',
                               qrcode=request.url_root + url_for("live",key=key)
                               )

@app.route('/about')
def about():
    return render_template('about.html', title = 'About Quantext')

@app.route('/help')
def help():
    return render_template('help.html', title = 'Help')

@app.route('/research')
def research():
    return render_template('research.html', title = 'Quantext Research')

def load_docs_from_disk(file_id,question_number,response_length):
    docs = []
    for i in range(0,response_length):
        origpath = os.path.join(app.config['DOC_FOLDER'],file_id,question_number,str(i))
        d = Doc(nlp.vocab).from_disk(origpath)
        docs.append(d)
    return docs

def update_setting(cuid,analysis_id,file_id,question_number,key,val):
    file = File.objects.get(id=file_id)
    analysis = Analysis.objects.get(id=analysis_id)
    analysis['files'][file_id]["columns"][question_number]['settings'][key] = val
    analysis.save()

    top_features, question = reprocessFeatures(cuid,analysis_id,file_id,question_number)            
    return render_template('keywords.html', analysis=analysis,file=file,question=question,features=top_features,current_user=current_user)

@app.route('/save_question_text', methods = ['POST'])
def save_question_text():
    file_id = request.get_json()['file_id']
    question_number = request.get_json()['question_number']
    new_text = request.get_json()['new_text']

    file = File.objects.get(id=file_id)
    idx = int(question_number)
    file["columns"][idx] = new_text
    file.save()
    return "Success"

@app.route('/settings', methods = ['POST'])
def settings():
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        analysis_id = request.get_json()['analysis_id']
        file_id = request.get_json()['file_id']
        question_number = request.get_json()['question_number']
        key = str(request.get_json()['key'])
        val = str(request.get_json()['val'])
        if not analysis_id is None and not file_id is None and not question_number is None:
            return update_setting(cuid,analysis_id,file_id,question_number,key,val)

@app.route('/label_settings', methods = ['POST'])
def label_settings():
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        analysis_id = request.get_json()['analysis_id']
        file_id = request.get_json()['file_id']
        question_number = request.get_json()['question_number']
        col = request.get_json()['col']
        key = str(request.get_json()['key'])
        val = str(request.get_json()['val'])
        if not analysis_id is None and not file_id is None and not question_number is None:
            analysis = Analysis.objects.get(id=analysis_id)
            analysis['files'][file_id]["columns"][question_number]['label_settings'][col][key] = val
            analysis.save()
            return "Success"

@celery.task(bind=True)
def search_and_sort(self,analysis_id,file_id,question_number,search_term,sort,labeller,cat_col):
    curr_percent = 1
    self.update_state(state='PROGRESS',meta={'current_task':'Starting search','process_percent':curr_percent})

    file = File.objects.get(id=file_id)
    analysis = Analysis.objects.get(id=analysis_id)
    idx = int(question_number)
    origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
    colpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_'+question_number+'.pkl')

    df = nlputility.read_pickle(origpath)
    col_df = nlputility.read_pickle(colpath)
    col = df.columns[idx]
    merge_df = nlputility.merge(df,col_df)
    merge_df = merge_df.filter(items=[col,'Characters','Words','Sentences','POS',cat_col,'LD','TTR','SMOG','Gunning','Flesch','FK','MTLD','HDD']).dropna(axis=0)

    kwics = None
    curr_percent = 15
    self.update_state(state='PROGRESS',meta={'current_task':'Searching','process_percent':curr_percent})

    #sort the filtered dataframe
    if not sort is None and not sort == "":
        sort_bits = sort.split('_', 1)
        asc = sort_bits[0] == "asc" #get whether ascending or descending

        if sort_bits[1] == 'Words' or sort_bits[1] == 'Sentences':
            merge_df = merge_df.sort_values(by=[sort_bits[1]],ascending=asc)
        else:
            c = sort_bits[1]
            if df[c].dtype == np.float64 or df[c].dtype == np.int64:
                df = df.sort_values(by=[c],ascending=asc)
                merge_df = merge_df.reindex(df.index).dropna(axis=0)
            else:
                df = df.loc[df[c].str.lower().sort_values(ascending=asc).index]
                merge_df = merge_df.reindex(df.index).dropna(axis=0)

    #search for rows that contain the search term
    if not search_term is None and not search_term == "":        
        kwics = return_kwics_from_pd(merge_df,search_term,col)

    curr_percent = 25
    self.update_state(state='PROGRESS',meta={'current_task':'Searching','process_percent':curr_percent})

    #collapse the dataframe, keeping only the relevant columns/rows to this column
    curr_percent = 50
    self.update_state(state='PROGRESS',meta={'current_task':'Searching','process_percent':curr_percent})

    #finally, return the list of tuples for the view
    tuples = list(merge_df.itertuples(index=True, name=None))

    question = {"text":df.columns[idx],"number":question_number}

    curr_percent = 75
    self.update_state(state='PROGRESS',meta={'current_task':'Searching','process_percent':curr_percent})

    if labeller:
        arr = analysis['files'][file_id]["columns"][question_number]["categories"]["all"]
        arr2 = [str(x) for x in sorted([int(y) for y in arr])]
        arr3 = {}
        
        for i in arr2:
            arr3[i] = arr[i]

        analysis['files'][file_id]["columns"][question_number]["categories"]["all"] = arr3

        responses_labeller = render_template('responses_labeller.html',
                                                analysis=analysis,
                                                file=file,
                                                question=question,
                                                tuples=tuples,
                                                cat_col=cat_col)
        rendered = {"responses_labeller":responses_labeller}
    else:
        responses_pos = render_template('responses_pos.html',
                           analysis=analysis,
                           file=file,
                           question=question,
                           tuples=tuples,
                           current_user=current_user,
                           kwics=kwics
                           )
        responses_kwic = render_template('responses_kwic.html',
                           analysis=analysis,
                           file=file,
                           question=question,
                           tuples=tuples,
                           current_user=current_user,
                           kwics=kwics
                           )
        rendered = {"responses_pos":responses_pos,"responses_kwic":responses_kwic}
    curr_percent = 100
    self.update_state(state='SUCCESS',meta={'current_task':'Finished searching','process_percent':curr_percent})
    return rendered

def return_kwics_from_pd(merge_df,search_term,col,ignoreboundaries=False,window_width=50):
    kwic_array = []

    if ignoreboundaries:
        pattern = r'%s' % re.escape(search_term.lower())
    else:
        pattern = r'\b%s\b' % re.escape(search_term.lower())
    
    search_results = merge_df[merge_df[col].str.contains(pattern, case=False, regex=True).fillna(False)]
    kwics = list(search_results.itertuples(index=True, name=None))

    for r in kwics:
        k1 = kwic(r[1].lower(), pattern, window_width)
        try:            
            if len(k1) > 0:
                k1[0].append(r[1])
                kwic_array.append((r[0],k1,r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],r[14]))
        except:
            print("nothing??")        
       
    #kwics = [(t[0],ks[idx],t[2],t[3],t[4],t[5],t[6],t[7],t[8],t[9],t[10],t[11],t[12],t[13],t[14]) for idx,t in enumerate(kwics)]
    return kwic_array

def formatline(l, window_width):
    return [l[0].rjust(window_width),l[1],l[2].ljust(window_width)]

def kwic(text,pattern, window_width):
    return [[text[max(0, match.start() - window_width): match.start()], match.group(),text[match.end(): match.end() + window_width]] for match in re.finditer(pattern, text)]

def get_kwic (responses, pattern, ignoreboundaries=False, window_width=50):
    kwic_array = []

    #if ignoreboundaries:
    #    pattern = keytext
    #else:
    #    pattern = r'\b'+keytext+r'\b'

    for r in responses:
        k1 = kwic(r.lower(), pattern, window_width)
        try:
            k1[0].append(r)
            kwic_array.append(k1)
        except:
            print("nothing??")        

    return kwic_array

def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return rv

@app.route('/studentFile_ajax/<analysis_id>/<file_id>')
def studentFile_ajax(analysis_id=None,file_id=None):
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        if not analysis_id is None and not file_id is None:
            analysis = Analysis.objects.get(id=analysis_id)
            file = File.objects.get(id=file_id)
            r_key = cuid + "_" + file_id
            nlp_obj = pickle.loads(r.get(r_key))

            origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
            df = nlputility.read_pickle(origpath)
            columns_to_analyse = sorted(analysis['files'][file_id]["columns"])
            new_columns, all_questions, histograms, total_means, total_urls, total_WS = [], [], [], [], [], []

            for c in columns_to_analyse:
                colpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_'+c+'.pkl')
                col_df = nlputility.read_pickle(colpath)
                new_columns.append({"text":file["columns"][int(c)],"number":c})

                u = URL.objects.get(analysis_id=analysis_id,file_id=file_id,question_number=c)
                total_urls.append(u.key)
                meansObj, histObj = {}, {}

                for w in ['Words','Sentences','LD','TTR','SMOG','Gunning','Flesch','FK','MTLD','HDD']:
                    histObj[w] = createChart(col_df[w].dropna().tolist(),False)
                    meansObj[w] = nlputility.get_mean(col_df[w].dropna().tolist())

                WS = {}
                WS['Words_total'] = col_df['Words'].sum()
                WS['Sentences_total'] = col_df['Sentences'].sum()

                WS['Words_min'] = col_df['Words'].min()
                WS['Sentences_min'] = col_df['Sentences'].min()

                WS['Words_max'] = col_df['Words'].max()
                WS['Sentences_max'] = col_df['Sentences'].max()

                total_WS.append(WS)
                total_means.append(meansObj)
                histograms.append(histObj)

                all_questions.append(df[df.columns[int(c)]].dropna().tolist())

                arr = analysis['files'][file_id]["columns"][c]["categories"]["all"]
                arr2 = [str(x) for x in sorted([int(y) for y in arr])]
                arr3 = {}
                
                for i in arr2:
                    arr3[i] = arr[i]

                analysis['files'][file_id]["columns"][c]["categories"]["all"] = arr3

            top_features = nlp_obj["top_features"]            

            return flask_response(stream_with_context(stream_template('studentFile.html',
                                                                      total_urls=total_urls,
                                                                      total_WS=total_WS,
                                                                      analysis=analysis,
                                                                      file=file,
                                                                      total_means=total_means,
                                                                      histograms=histograms,
                                                                      top_features=top_features,
                                                                      columns=new_columns,
                                                                      responses=all_questions,
                                                                      cat_col="Categories_1"                                                                      
                                                                      )))

@app.route('/responses_only/<analysis_id>/<file_id>/<question_number>/<t>')
def responses_only(analysis_id=None,file_id=None,question_number=None,t=None):
    if not current_user.is_anonymous:
        if not analysis_id is None and not file_id is None:
            analysis = Analysis.objects.get(id=analysis_id)
            file = File.objects.get(id=file_id)
            question = {"number":question_number}

            idx = int(question_number)
            origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
            colpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_'+question_number+'.pkl')

            df = nlputility.read_pickle(origpath)
            col_df = nlputility.read_pickle(colpath)
            merge_df = nlputility.merge(df,col_df)

            col = df.columns[idx]
            merge_df = merge_df.filter(items=[col,'Characters','Words','Sentences','POS','Categories_1','LD','TTR','SMOG','Gunning','Flesch','FK','MTLD','HDD']).dropna(axis=0)
            tuples = list(merge_df.itertuples(index=True, name=None))

            if t == "pos":
                return render_template('responses_pos.html',analysis=analysis,file=file,question=question,tuples=tuples,cat_col="Categories_1")
            elif t == "labeller":
                return render_template('responses_labeller.html',analysis=analysis,file=file,question=question,tuples=tuples,cat_col="Categories_1")
    
    return "Error"

@app.route('/single_file/<file_id>', methods=['GET'])
def single_file(file_id=None):
    if not file_id is None:
        file = File.objects.get(id=file_id)
        new_f = render_template('partials/file_new_analysis.html',file=file)
        f = render_template('partials/file.html',file=file)
        return json.dumps({"file":f,"new_file":new_f})

@app.route('/analyses', methods=['GET'])
def analyses():
    if not current_user is None:
        analyses = Analysis.objects(owner=current_user['id']).exclude('files.questions').exclude('files.text').exclude('files.statistics').order_by('-created')
        if len(analyses) > 0:
            return render_template('partials/analyses.html',analyses=analyses)
        else:
            return render_template("partials/empty.html", list="analyses")

@app.route('/shared_with_me', methods=['GET'])
def shared_with_me():
    if not current_user is None:
        shared_list = Analysis.objects(shared__contains=current_user['id'])
        if len(shared_list) > 0:
            return render_template('partials/analyses.html',analyses=shared_list)
        else:
            return render_template("partials/empty.html", list="shared")

#API methods
#@app.route('/indices/<analysis_id>/<file_id>/<question_number>', methods=['GET'])
#def indices(analysis_id=None, file_id=None, question_number=None):
#    if not current_user.is_anonymous:
#        if not analysis_id is None and not file_id is None and not question_number is None:
#            job = get_indices_task.delay(analysis_id,file_id,question_number)
#            json_data = {"job_id":job.id}
#            return json.dumps(json_data), 200
'''
@celery.task(bind=True)
def get_indices_task(self, analysis_id, file_id, question_number):
    self.update_state(state='PROGRESS',meta={'current_task':'Fetching indices','process_percent':1})
    analysis = Analysis.objects.get(id=analysis_id)
    file = File.objects.get(id=file_id)
    idx = int(question_number)
    self.update_state(state='PROGRESS',meta={'current_task':'Fetching indices','process_percent':20})

    origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id, analysis_id+'_data.pkl')
    df = nlputility.read_pickle(origpath)
    columns = df.columns
    col = columns[idx]
    question = {"text":col,"number":question_number}

    total_indices = []
    docs = load_docs_from_disk(file_id,question_number,len(df[col].dropna().tolist()))
    self.update_state(state='PROGRESS',meta={'current_task':'Fetching indices','process_percent':60})
    curr_percent = 60
    step = 20/len(docs)

    for doc in docs:
        curr_percent += step
        self.update_state(state='PROGRESS',meta={'current_task':'Fetching indices','process_percent':curr_percent})
        i = get_indices(doc,analysis['files'][file_id]["columns"][question_number]['settings'])
        total_indices.append(i)

    means = nlputility.get_indices_means(total_indices)
    rendered = render_template('indices.html',file=file,question=question,means=means)

    self.update_state(state='SUCCESS',meta={'current_task':'Finished searching','process_percent':100})
    return rendered
'''

def subset_traverser(kwics, word):
    obj = {}
    obj[word] = {"count":0,"children":{}}
    total_levels = 0
    for k in kwics:
        if len(k) > 0:
            obj[word]["count"] += 1
            right_side = k[0][2]
            curr = obj[word]["children"]
            curr_levels = 0
            for w in right_side.split(" "):
                curr_levels += 1
                w = w.lower()
                if not w in curr:
                    curr[w] = {"count":1,"children":{}}
                else:
                    curr[w]["count"] += 1
                curr = curr[w]["children"]
            if curr_levels > total_levels:
                total_levels = curr_levels

    obj["total_levels"] = total_levels
    return obj

@app.route('/viz/<analysis_id>/<file_id>/<question_number>', methods=['GET'])
def viz(analysis_id=None,file_id=None, question_number=None):
    if not current_user.is_anonymous:
        if not file_id is None and not question_number is None:
            idx = int(question_number)

            origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
            df = nlputility.read_pickle(origpath)
            responses = df[df.columns[idx]].dropna().tolist()
            srch = request.args.get("search")

            pattern = r'%s' % re.escape(srch.lower())

            if srch[0] == "\"" and srch[len(srch) - 1] == "\"":
                srch = srch[1:-1]
                pattern = r'\b%s\b' % re.escape(srch.lower())
            elif '|' in srch:
                searches = srch.split("|")
                patterns = [r'\b%s\b' % re.escape(s.lower()) for s in searches]
                pattern = re.compile('|'.join(patterns))
            elif '+' in srch:
                searches = srch.split("+")
                srch = ' '.join(searches)
                pattern = r'\b%s\b' % re.escape(srch.lower())

            selectors = [re.search(pattern,x.lower()) for x in responses]
            responses = list(itertools.compress(responses, selectors))
            responses = get_kwic(responses, pattern)

            subset = subset_traverser(responses,srch.lower())
            return json.dumps(subset)

@app.route('/responses_pos/<analysis_id>/<file_id>/<question_number>', methods=['GET'])
def responses_pos(analysis_id=None,file_id=None, question_number=None):
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        if not file_id is None and not question_number is None:
            idx = int(question_number)

            origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
            df = nlputility.read_pickle(origpath)
            responses = df[df.columns[idx]].dropna().tolist()

            r_key = cuid + "_" + file_id
            nlp_obj = pickle.loads(r.get(r_key))
            statistics = nlp_obj["statistics"][question_number]

            if request.args.get('format') == 'json':
                return json.dumps([f['POS'] for f in statistics])
            elif request.args.get('format') == 'view':
                return flask_response(stream_with_context(stream_template('responses_pos.html',
                                                                          responses=responses,
                                                                          statistics=statistics)))
            else:
                return json.dumps([f['POS'] for f in statistics])

@app.route('/collocation_search/<analysis_id>/<file_id>/<question_number>', methods=['GET'])
def collocation_search(analysis_id=None,file_id=None,question_number=None):
    if not current_user.is_anonymous:
        if not analysis_id is None and not file_id is None and not question_number is None:
            search = request.args.get("search")
            job = collocation_search_task.delay(analysis_id,file_id,question_number,search)
            json_data = {"job_id":job.id}
            return json.dumps(json_data), 200

@app.route('/reset_collocation/<analysis_id>/<file_id>/<question_number>', methods=['GET'])
def reset_collocation(analysis_id=None,file_id=None,question_number=None):
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        if not analysis_id is None and not file_id is None and not question_number is None:
            analysis = Analysis.objects.get(id=analysis_id)
            file = File.objects.get(id=file_id)
            idx = int(question_number)

            origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
            df = nlputility.read_pickle(origpath)
            question = {"text":df.columns[idx],"number":question_number}

            r_key = cuid + "_" + file_id
            nlp_obj = pickle.loads(r.get(r_key))
            features = nlp_obj["top_features"][question_number]
            return render_template('keywords.html',analysis=analysis,file=file,question=question,features=features, current_user=current_user)

@celery.task(bind=True)
def collocation_search_task(self,analysis_id,file_id,question_number,srch):
    self.update_state(state='PROGRESS',meta={'current_task':'Starting search','process_percent':1})
    analysis = Analysis.objects.get(id=analysis_id)
    file = File.objects.get(id=file_id)
    idx = int(question_number)

    origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
    df = nlputility.read_pickle(origpath)
    self.update_state(state='PROGRESS',meta={'current_task':'Starting search','process_percent':10})

    processed_settings = process_settings(analysis['files'][file_id]["columns"][question_number]["settings"])
    self.update_state(state='PROGRESS',meta={'current_task':'Searching','process_percent':50})

    docs = load_docs_from_disk(file_id,question_number,len(df[df.columns[idx]].dropna().tolist()))
    self.update_state(state='PROGRESS',meta={'current_task':'Searching','process_percent':65})

    feats = get_features(docs,processed_settings,srch)
    self.update_state(state='PROGRESS',meta={'current_task':'Searching','process_percent':75})

    top_features = get_top_features(feats,processed_settings)
    self.update_state(state='PROGRESS',meta={'current_task':'Searching','process_percent':85})

    question = {"text":df.columns[idx],"number":question_number}
    rendered = render_template('keywords.html',analysis=analysis,file=file,question=question,features=top_features, current_user=current_user)

    self.update_state(state='SUCCESS',meta={'current_task':'Finished searching','process_percent':100})
    return rendered

@app.route('/poll_search', methods =['POST'])
def poll_search():
    task_id = request.get_json()['task_id']
    task = search_and_sort.AsyncResult(task_id)
    data = task.result
    return json.dumps(data)

@app.route('/poll_collocation_search', methods =['POST'])
def poll_collocation_search():
    task_id = request.get_json()['task_id']
    task = collocation_search_task.AsyncResult(task_id)
    data = task.result
    return json.dumps(data)

@app.route('/poll_indices', methods =['POST'])
def poll_indices():
    task_id = request.get_json()['task_id']
    task = get_indices_task.AsyncResult(task_id)
    data = task.result
    return json.dumps(data)

@app.route('/search/<analysis_id>/<file_id>/<question_number>', methods=['GET'])
def search(analysis_id=None,file_id=None,question_number=None):
    if not current_user.is_anonymous:
        if not analysis_id is None and not file_id is None and not question_number is None:
            search_term = request.args.get("search_term")
            sort = request.args.get("sort")
            labeller = request.args.get("labeller")
            labeller = labeller == True or labeller == 'True'
            cat_col = request.args.get("cat_col")

            job = search_and_sort.delay(analysis_id,file_id,question_number,search_term,sort,labeller,cat_col)
            json_data = {"job_id":job.id}
            return json.dumps(json_data), 200

@app.route('/new_set', methods=['POST'])
def new_set():
    analysis_id = request.get_json()['analysis_id']
    file_id = request.get_json()['file_id']
    question_number = request.get_json()['question_number']
    name = request.get_json()['name']

    if not analysis_id is None and not file_id is None and not question_number is None:
        colpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+"_"+question_number+".pkl")
        col_df = nlputility.read_pickle(colpath)
        cat_cols = sorted([col for col in col_df.columns if 'Categories_' in col])

        last = cat_cols[-1].split("_")
        last = int(last[1])
        last = last + 1

        col = 'Categories_'+str(last)
        col_df[col] = [[]]*len(col_df)
        col_df.to_pickle(colpath)

        analysis = Analysis.objects.get(id=analysis_id)
        analysis['files'][file_id]["columns"][question_number]["categories"][col] = {}
        analysis['files'][file_id]["columns"][question_number]["settings"]["aliases"][col] = name
        analysis['files'][file_id]["columns"][question_number]["label_settings"][col] = {"randomise_labels":False,"hide_labels":False}

        analysis.save()

        return "Success"
    
@app.route('/new_label', methods=['POST'])
def new_label():
    analysis_id = request.get_json()['analysis_id']
    file_id = request.get_json()['file_id']
    question_number = request.get_json()['question_number']
    label = request.get_json()['label']
    cat_col = request.get_json()['cat_col']
    cat_col = "Categories_"+cat_col

    if not analysis_id is None and not file_id is None and not question_number is None:
        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        keys = list(analysis['files'][file_id]["columns"][question_number]["categories"]["all"].keys())
        if len(keys) > 0:
            new_key = 0
            for key in keys:    
                key = int(key)            
                if key > new_key:
                    new_key = key
            new_key += 1
        else:
            new_key = 0
        analysis['files'][file_id]["columns"][question_number]["categories"]["all"][str(new_key)] = {"label":label,"used":0}
        analysis.save()
        question = {"number":question_number}

        idx = int(question_number)
        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_data.pkl")
        df = nlputility.read_pickle(pickle_path)
        col = df.columns[idx]

        aliases = analysis['files'][file_id]["columns"][question_number]["settings"]["aliases"]
        colpath = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_"+question_number+".pkl")
        col_df = nlputility.read_pickle(colpath)
        cat_cols = sorted([col for col in col_df.columns if 'Categories_' in col])

        col_df = col_df.filter(items=cat_cols).dropna(axis=0)
        total = len(df[col].dropna().tolist())
        percents, marked_up = [], []
        
        for c in col_df.columns:
            m = len([x for x in col_df[c].dropna().tolist() if len(x) != 0])
            percent = m/total*100
            marked_up.append(m)
            percents.append(percent)

        url = URL.objects.get(analysis_id=analysis_id,file_id=file_id,question_number=question_number)
        key = url["key"]

        arr = analysis['files'][file_id]["columns"][question_number]["categories"]["all"]
        arr2 = [str(x) for x in sorted([int(y) for y in arr])]
        arr3 = {}
        
        for i in arr2:
            arr3[i] = arr[i]

        analysis['files'][file_id]["columns"][question_number]["categories"]["all"] = arr3

        custom_word_list = render_template("custom_word_list.html", analysis=analysis, file=file, question=question)
        histograms = render_template("set_histograms.html", key=key, analysis=analysis, file=file, question=question, aliases=aliases, percents=percents, cat_cols=cat_cols, alpha=alpha)
        master_label = render_template("master_label.html",analysis=analysis,file=file,question=question,label_key=new_key)
        category = render_template("category.html",key=new_key,text=label,length=len(list(analysis['files'][file_id]["columns"][question_number]["categories"]["all"].keys())))
        label_bars = render_template("label_bars.html",analysis=analysis,file=file,question=question,cat_col=cat_col)
        return json.dumps({"master_label":master_label,"category":category,"histograms":histograms,"label_bars":label_bars})

@app.route('/update_label', methods=['POST'])
def update_label():
    analysis_id = request.get_json()['analysis_id']
    file_id = request.get_json()['file_id']
    question_number = request.get_json()['question_number']
    label = request.get_json()['label']
    key = request.get_json()['key']
    cat_col = request.get_json()['cat_col']
    cat_col = "Categories_"+cat_col

    if not analysis_id is None and not file_id is None and not question_number is None:
        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        
        analysis['files'][file_id]["columns"][question_number]["categories"]["all"][key]['label'] = label
        analysis.save()
        question = {"number":question_number}

        idx = int(question_number)
        pickle_path = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_data.pkl")
        df = nlputility.read_pickle(pickle_path)
        col = df.columns[idx]

        aliases = analysis['files'][file_id]["columns"][question_number]["settings"]["aliases"]
        colpath = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_"+question_number+".pkl")
        col_df = nlputility.read_pickle(colpath)
        cat_cols = sorted([col for col in col_df.columns if 'Categories_' in col])

        col_df = col_df.filter(items=cat_cols).dropna(axis=0)
        total = len(df[col].dropna().tolist())
        percents, marked_up = [], []
        
        for c in col_df.columns:
            m = len([x for x in col_df[c].dropna().tolist() if len(x) != 0])
            percent = m/total*100
            marked_up.append(m)
            percents.append(percent)

        url = URL.objects.get(analysis_id=analysis_id,file_id=file_id,question_number=question_number)
        key = url["key"]

        arr = analysis['files'][file_id]["columns"][question_number]["categories"]["all"]
        arr2 = [str(x) for x in sorted([int(y) for y in arr])]
        arr3 = {}
        
        for i in arr2:
            arr3[i] = arr[i]

        analysis['files'][file_id]["columns"][question_number]["categories"]["all"] = arr3

        histograms = render_template("set_histograms.html",key=key,analysis=analysis, file=file, question=question,aliases=aliases,percents=percents,cat_cols=cat_cols,alpha=alpha)
        label_bars = render_template("label_bars.html",analysis=analysis,file=file,question=question,cat_col=cat_col)

        return json.dumps({"histograms":histograms,"label_bars":label_bars})

@app.route('/delete_label', methods=['POST'])
def delete_label():
    analysis_id = request.get_json()['analysis_id']
    file_id = request.get_json()['file_id']
    question_number = request.get_json()['question_number']
    key = request.get_json()['key']

    if not analysis_id is None and not file_id is None and not question_number is None:
        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        analysis['files'][file_id]["columns"][question_number]["categories"]["all"].pop(key)

        colpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+"_"+question_number+".pkl")
        col_df = nlputility.read_pickle(colpath)
        cat_cols = sorted([col for col in col_df.columns if 'Categories_' in col])

        for cat_col in cat_cols:
            if key in analysis['files'][file_id]["columns"][question_number]["categories"][cat_col].keys():
                analysis['files'][file_id]["columns"][question_number]["categories"][cat_col].pop(key)

            for index, row in col_df.iterrows():                
                arr = [cat for cat in list(col_df.at[index,cat_col]) if cat != key]
                col_df.at[index,cat_col] = arr  
        
        col_df.to_pickle(colpath)
        analysis.save()          
        return "Success"

@app.route('/label_response', methods=['POST'])
def label_response():
    id = request.get_json()['id']
    analysis_id = request.get_json()['analysis_id']
    file_id = request.get_json()['file_id']
    question_number = request.get_json()['question_number']
    selected = request.get_json()['selected']
    category = request.get_json()['category']
    cat_col = request.get_json()['cat_col']
    cat_col = "Categories_"+cat_col

    if not analysis_id is None and not file_id is None and not question_number is None:
        analysis = Analysis.objects.get(id=analysis_id)
        origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_'+question_number+'.pkl')
        col_df = nlputility.read_pickle(origpath)
        if category not in analysis['files'][file_id]["columns"][question_number]["categories"][cat_col].keys():
            analysis['files'][file_id]["columns"][question_number]["categories"][cat_col][category] = 0

        if str(selected) == "True" or str(selected) == "true":
            analysis['files'][file_id]["columns"][question_number]["categories"]["all"][category]["used"]+=1
            analysis['files'][file_id]["columns"][question_number]["categories"][cat_col][category]+=1
            arr = list(col_df.at[int(id),cat_col])
            arr.append(category)
            col_df.at[int(id),cat_col] = arr
        else:
            arr = [cat for cat in list(col_df.at[int(id),cat_col]) if cat != category]
            col_df.at[int(id),cat_col] = arr
            analysis['files'][file_id]["columns"][question_number]["categories"]["all"][category]["used"]-=1
            analysis['files'][file_id]["columns"][question_number]["categories"][cat_col][category]-=1

        col_df.to_pickle(origpath)
        analysis.save()
        return "Success"

@app.route('/update_live_settings', methods=['POST'])
def update_live_settings():
    if not current_user.is_anonymous:
        file_id = request.get_json()['file_id']
        analysis_id = request.get_json()['analysis_id']
        name = request.get_json()['name']
        val = request.get_json()['val']

        analysis = Analysis.objects.get(id=analysis_id)
        analysis['files'][file_id]["columns"]["0"]["settings"][name] = val
        analysis.save()

        return "Success"

@app.route('/close_help', methods=['POST'])
def close_help():
    if not current_user.is_anonymous:
        help = request.get_json()['help']

        current_user["config"][help] = "False"
        current_user.save()

        return "Success"

def reprocessFeatures(cuid,analysis_id,file_id,question_number):
    analysis = Analysis.objects.get(id=analysis_id)
    idx = int(question_number)
    origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
    df = nlputility.read_pickle(origpath)
    responses = df[df.columns[idx]].dropna().tolist()
    processed_settings = process_settings(analysis['files'][file_id]["columns"][question_number]['settings'])
    docs = load_docs_from_disk(file_id,question_number,len(responses))
    top_features = get_top_features(get_features(docs,processed_settings,None),processed_settings)
    question = {"text":df.columns[idx],"number":question_number}

    r_key = cuid + "_" + file_id
    nlp_obj = pickle.loads(r.get(r_key))
    nlp_obj["top_features"][question_number] = top_features
    r.set(r_key,pickle.dumps(nlp_obj))
    return top_features, question

@app.route('/exclude_word', methods=['POST'])
def exclude_word():
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        analysis_id = request.get_json()['analysis_id']
        file_id = request.get_json()['file_id']
        question_number = request.get_json()['question_number']
        word = request.get_json()['word']

        if not analysis_id is None and not file_id is None and not question_number is None:
            file = File.objects.get(id=file_id)
            analysis = Analysis.objects.get(id=analysis_id)
            blist = analysis['files'][file_id]["columns"][question_number]['settings']['blacklist']
            if word not in blist:
                blist.append(word)
            analysis['files'][file_id]["columns"][question_number]['settings']['blacklist'] = blist
            analysis.save()

            top_features, question = reprocessFeatures(cuid,analysis_id,file_id,question_number)
            keywords = render_template('keywords.html', analysis=analysis,file=file,question=question,features=top_features,current_user=current_user)
            question_settings = render_template('question_settings.html',analysis=analysis,file=file,question=question)
            return json.dumps({"keywords":keywords,"question_settings":question_settings})

@app.route('/save_blist', methods=['POST'])
def save_blist():
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        analysis_id = request.get_json()['analysis_id']
        file_id = request.get_json()['file_id']
        question_number = request.get_json()['question_number']
        custom = request.get_json()['custom']
        custom = custom.split(",")

        if not analysis_id is None and not file_id is None and not question_number is None:
            file = File.objects.get(id=file_id)
            analysis = Analysis.objects.get(id=analysis_id)
            analysis['files'][file_id]["columns"][question_number]['settings']['blacklist'] = custom
            analysis.save()

            top_features, question = reprocessFeatures(cuid,analysis_id,file_id,question_number)
            keywords = render_template('keywords.html', analysis=analysis,file=file,question=question,features=top_features,current_user=current_user)
            question_settings = render_template('question_settings.html',analysis=analysis,file=file,question=question)
            return json.dumps({"keywords":keywords,"question_settings":question_settings})          

@app.route('/save_wlist', methods=['POST'])
def save_wlist():
    if not current_user.is_anonymous:
        cuid = str(current_user['id'])
        analysis_id = request.get_json()['analysis_id']
        file_id = request.get_json()['file_id']
        question_number = request.get_json()['question_number']
        custom = request.get_json()['custom']
        custom = custom.split(",")

        if not analysis_id is None and not file_id is None and not question_number is None:
            file = File.objects.get(id=file_id)
            analysis = Analysis.objects.get(id=analysis_id)            
            analysis['files'][file_id]["columns"][question_number]['settings']['whitelist'] = custom
            analysis.save()

            top_features, question = reprocessFeatures(cuid,analysis_id,file_id,question_number)
            keywords = render_template('keywords.html', analysis=analysis,file=file,question=question,features=top_features,current_user=current_user)
            question_settings = render_template('question_settings.html',analysis=analysis,file=file,question=question)
            return json.dumps({"keywords":keywords,"question_settings":question_settings})    

@app.route('/generate_report', methods=['POST'])
def report():
    if not current_user.is_anonymous:
        analysis_id = request.get_json()['analysis_id']
        file_id = request.get_json()['file_id']
        questions = request.get_json()['questions']
        nkey_report = request.get_json()['nkey_report']
        nkey_report = int(nkey_report)
        questions = json.loads(questions)
        question_list = []

        if not analysis_id is None:
            analysis = Analysis.objects.get(id=analysis_id)
            if not file_id is None:
                origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
                df = nlputility.read_pickle(origpath)
                for q in questions:
                    colpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_'+q['question_number']+'.pkl')
                    col_df = nlputility.read_pickle(colpath)
                    mean_words = "{0:.0f}".format(nlputility.get_mean(col_df['Words'].dropna().tolist()))
                    mean_sentences = "{0:.0f}".format(nlputility.get_mean(col_df['Sentences'].dropna().tolist()))

                    col_df = col_df.filter(items=["Categories_1"]).dropna(axis=0)
                    idx = int(q["question_number"])
                    col = df.columns[idx]
                    subset = df.filter(items=[col]).dropna(axis=0)

                    an_cats = analysis['files'][file_id]["columns"][q["question_number"]]["categories"]["all"]
                    cats = {}
                    for k in an_cats:
                        cats[k] = []

                    for c in an_cats:
                        for res in col_df.itertuples():
                            if c in res[1]:
                                cats[c].append(subset.at[res[0],col])

                    total_indices = []
                    docs = load_docs_from_disk(file_id,str(q['question_number']),len(subset[col].tolist()))
                    for doc in docs:
                        i = get_indices(doc,analysis['files'][file_id]["columns"][q['question_number']]['settings'])
                        total_indices.append(i)
                    means = nlputility.get_indices_means(total_indices)

                    cat_max = 0
                    cat_array = []
                    for k,v in cats.items():
                        cat_max = len(v) if len(v) > cat_max else cat_max
                        cat_array.append([analysis['files'][file_id]["columns"][q["question_number"]]["categories"]["all"][k]['label'],len(v)])

                  #  histograms = createCharts(nlp_obj["statistics"][idx],['Words','Sentences'],[False,False])
                    processed_settings = process_settings(analysis['files'][file_id]["columns"][q['question_number']]['settings'])
                    processed_settings["nkey"] = nkey_report
                    top_features = get_top_features(get_features(docs,processed_settings,None),processed_settings) if "most_frequent" in q else {}

                    question_obj = {
                                        "question_number":q["question_number"],
                                        "question_title":col,
                                        "question_text":col,
                                        "top_features":top_features,
                                        "statistics":{'Words':mean_words,'Sentences':mean_sentences} if "summary_statistics" in q else {},
                                        "responses":len(subset.index),
                                        "categories":cat_array,
                                        "cat_max":cat_max,
                                        "examples":[[analysis['files'][file_id]["columns"][q["question_number"]]["categories"]["all"][k]['label'],v] for k,v in cats.items()],
                                        "indices":means if "indices" in q else {}
                                        #"histograms":histograms
                                    }
                    question_list.append(question_obj)
    return json.dumps(question_list)

@app.route('/export_xlsx', methods=['POST'])
def export_xlsx():
    analysis_id = request.get_json()['analysis_id']
    file_id = request.get_json()['file_id']
    include_labels = request.get_json()["include_labels"]

    if not analysis_id is None and not file_id is None:
        analysis = Analysis.objects.get(id=analysis_id)
        file = File.objects.get(id=file_id)
        origpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+'_data.pkl')
        df = nlputility.read_pickle(origpath)
        columns_to_analyse = analysis['files'][file_id]["columns"]
        merge_df = df

        if include_labels == "true":
            for i in sorted(columns_to_analyse.keys()):
                colpath = os.path.join(app.config['PANDAS_FOLDER'],str(file['id']),analysis_id+"_"+str(i)+".pkl")
                col_df = nlputility.read_pickle(colpath)
                col_df = col_df.filter(items=['Categories_1'])

                lst = col_df["Categories_1"]

                for l in columns_to_analyse[i]["categories"]["all"].keys():
                    for xx,row in enumerate(lst):          
                        row = list(row)
                        if l in row:            
                            for ii,item in enumerate(row):
                                if item == l:
                                    lst[xx][ii] = columns_to_analyse[i]["categories"]["all"][l]['label']
            
            col_df["Categories_1"] = lst
            merge_df = nlputility.merge(merge_df,col_df)

        excelpath = os.path.join(app.config['PANDAS_FOLDER'],file_id,analysis_id+"_output.xlsx")
        merge_df.to_excel(excelpath,index=False)

        return send_file(excelpath,attachment_filename=analysis_id+"_output.xlsx")

@app.route('/export_users', methods=['GET'])
def export_users():
    users = User.objects()
    users = [{'username':user['username'],'email':user['email'],'display_name':user['display_name']} for user in users]
    return json.dumps(users)