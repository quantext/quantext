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


import textacy
import re
from nltk import FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def count_ngrams(corpus, settings):
    warning = ""

    for k in settings.keys():
        if type(settings[k]) == str:
            if settings[k] == "True":
                settings[k] = True
            elif settings[k] == "False":
                settings[k] = False
            elif re.match(r'\d+',settings[k]):
                settings[k] = int(settings[k])
            else:
                break  
    
    nkey = settings['nkey']
    measure = settings['measure']
    kblack = settings['kblack']
    bblack = settings['bblack'] 
    punct = settings['punct']           
    nums = settings['nums']

    #Find all words and bigrams in all responses in corpus
    words = [list(textacy.extract.ngrams(doc,1,filter_stops=kblack, filter_punct=punct, filter_nums=nums)) for doc in corpus]
    bigrams = [list(textacy.extract.ngrams(doc,2,filter_stops=bblack, filter_punct=punct, filter_nums=nums)) for doc in corpus]
    #flatten the lists to calculate frequency distribution and get text from spacy span
    words_flat = [w.text for wds in words for w in wds] 
    word_fd = FreqDist(words_flat)
    top_keywords = [(w, word_fd[w]) for w in sorted(word_fd, key=word_fd.get, reverse=True)]
    
    bigram_measures = BigramAssocMeasures() 
    
    bigrams_flat = [b.text for bgs in bigrams for b in bgs] 
    # next lines just to get bigrams in the correct tuple format for nltk FreqDist and remove any ooops caused by choice of punct settings
    bigrams_flat = [tuple(s.split(" ")) for s in bigrams_flat]
    bigrams_flat = [b for b in bigrams_flat if len(b) > 1]
        
    bigram_fd = FreqDist(bigrams_flat)
    finder = BigramCollocationFinder(word_fd, bigram_fd)

    if measure == "LR":
        try:
            top_bigrams = finder.nbest(bigram_measures.likelihood_ratio, nkey)
        except:
            warning = "Problem with LR measure. Default to simple bigram frequency (RAW setting)"
            print(warning)
            top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)
    elif measure == "PMI":
        try:
            top_bigrams = finder.nbest(bigram_measures.pmi, nkey)
        except:
            warning = "Problem with PMI measure. Default to simple bigram frequency (RAW setting)"
            print(warning)
            top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)
    elif measure == "CHISQ":
        try:
            top_bigrams = finder.nbest(bigram_measures.chi_sq, nkey)
        except:
            warning = "Problem with CHISQ measure. Default to simple bigram frequency (RAW setting)"
            print(warning)
            top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)
    elif measure == "STUDT":
        try:
            top_bigrams = finder.nbest(bigram_measures.student_t, nkey)
        except:
            warning = "Problem with STUDT measure. Default to simple bigram frequency (RAW setting)"
            print(warning)
            top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)
    else:
        top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)

    #score bigrams using likelihood ratio but more helpful to end user to see raw counts + explain measure used in tool tip
    top_bg_with_count = sorted([(bg, count) for (bg, count) in finder.ngram_fd.items() if bg in top_bigrams], key=lambda bgcount:-bgcount[1])
    top_bigrams = [(bg, count) for (bg, count) in top_bg_with_count if count > 1 and bg[0]!=bg[1]]
    
    return top_keywords[:nkey],top_bigrams, warning  

def content_words_in_response(tokenised_response):
    content_words = 0
    tokenised_response = [t for tok in tokenised_response for t in tok] #flatten - ignore sentence breaks in list
    for t in tokenised_response:
        if t[1][:3] in ('ADJ','NOU','ADV','VER'): # count verbs, adjectives, adverbs and nouns as content words
                content_words +=1               
    return content_words  

def set_whitelist(corpus,wlist):            
    for w in corpus.spacy_vocab:
        w.is_stop = True
    for w in wlist:
        corpus.spacy_vocab[w].is_stop = False
    return 

def set_stoplist(corpus,settings):
    if settings['blist']:
        for b in settings['blist']:
            corpus.spacy_vocab[b].is_stop = True

    if settings['wlist']:
        for w in settings['wlist']:
            corpus.spacy_vocab[w].is_stop = True

    if settings['white'] == 'True' and settings['wlist']:
        for w in settings['wlist']:
            corpus.spacy_vocab[w].is_stop = False
    return

def analysecorpus(corpus, settings):
    set_stoplist(corpus,settings)
    return count_ngrams(corpus,settings)

def kwic(corpus,keyword_phrase):
    kwic = []
    for doc in corpus.docs:
        kwic.append(textacy.text_utils.KWIC(doc.text,keyword=keyword_phrase,print_only=False))
    return [l for line in kwic for l in line]
        
def get_words(doc):
    return list(textacy.extract.words(doc, filter_punct=True, filter_stops=False, filter_nums=False)) #Leave as default for worksheet calculations
        
def count_syllables(words):
    hyphenator = textacy.data.load_hyphenator(lang='en')
    syllables_per_word = [len(hyphenator.positions(word.lower_)) + 1 for word in words]
    n_syllables = sum(syllables_per_word)
    n_polysyllable_words = sum(1 for n in syllables_per_word if n >= 3)
    return n_syllables, n_polysyllable_words

def similaritytoselection(selection,corpus):
    #selection is any selected text string e.g. model answer or any response selected from student responses
    #returns a list of tuples in order from most to least similar to selected text (on a scale of 1 - 0) where tuple is (similarity, selectedtext)
    s = textacy.Doc(selection)
    simlist = [(textacy.similarity.word2vec(s,doc), doc.text) for doc in corpus]
    simlist.sort(key = lambda v:-v[0])
    return simlist
