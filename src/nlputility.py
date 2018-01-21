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

import re
import textacy
import networkx as nx
from networkx.readwrite import json_graph
from nltk import FreqDist
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from math import sqrt

def strip_punctuation(w,settings):
    punctuation = settings['punctuation']
    return ''.join(c for c in w if c not in punctuation)

def normalise_text(corpus,settings):
    ncontractions = settings['ncontractions']
    lcase = settings['lcase']
    punct = settings['punct']
    nums = settings['nums']

    textlist = [d.text for d in corpus.docs]

    if ncontractions:
        textlist = [textacy.preprocess.unpack_contractions(w) for w in textlist]
        textlist = [re.sub(r"(\b)(nt|n't)", r"not", w) for w in textlist] # hack to deal with standalone n't missing from textacy unpack_contractions after tokenisation

    if lcase:
        textlist = [w.lower() for w in textlist]
    if punct:
        textlist = [strip_punctuation(w,settings) for w in textlist]
    if nums:
        textlist = [textacy.preprocess.replace_numbers(w,"") for w in textlist]
    ncorpus = textacy.Corpus("en")
    for t in textlist:
        ncorpus.add_text(t)
    return ncorpus

def getwords(corpus,q):
    stops = q['qSettings']['kblack']
    words = [list(textacy.extract.ngrams(doc,1,filter_stops=stops)) for doc in corpus]
    if stops:
        words = [w.text for wds in words for w in wds if not w.text in q['blist']]  #flatten the list to calculate frequency distribution, get text from spacy span and remove blist words.
    else:
        words = [w.text for wds in words for w in wds]
    return words

def getngrams(ncorpus,q,n):
    #expects normalised corpus text
    stops = q['qSettings']['kblack']

    ngrams = [list(textacy.extract.ngrams(doc,n,filter_stops=stops)) for doc in ncorpus]
    ngrams = [ng.text for ngs in ngrams for ng in ngs]

    ngrams = [tuple(s.split(" ")) for s in ngrams]
    ngrams = [ng for ng in ngrams if len(ng) == n]
    if stops:
        ngrams = [ng for ng in ngrams if not True in [i in q['blist'] for i in ng]] #if *any* blist word in an ngram, do not return the ngram
    else:
        ngrams = [ng for ng in ngrams] #if *any* blist word in an ngram, do not return the ngram
    return ngrams

def buildcollocgraph(corpus,settings):
    u = nx.Graph()
    wordsall = []
    window = settings['window']
    stem = settings['stem']
    kblack = settings['kblack']
    cgcutoff = settings['cgcutoff']
    ncorpus = normalise_text(corpus,settings) #normalise corpus here
    for doc in ncorpus:
        words = [textacy.extract.ngrams(doc,1,filter_stops=kblack)]
        words = [t.text for word in words for t in word]

        if len(words) > cgcutoff:
            g = textacy.network.terms_to_semantic_network(words, normalize=stem, window_width=window, edge_weighting='cooc_freq')
            u.add_nodes_from(g.nodes(data=True))
            u.add_edges_from(g.edges(data=True))
            wordsall.append(words)
    wordsall = [w for wdlist in wordsall for w in wdlist]
    word_fd = FreqDist(wordsall)
    #test visualise
    #textacy.viz.network.draw_semantic_network(U, node_weights=word_fd, spread=3.0, draw_nodes=True, base_node_size=300, node_alpha=0.25, line_width=0.5, line_alpha=0.1, base_font_size=12, save=False)
    #convert networkx graph to json for d3
    for i,v in [k for k in word_fd.items()]:
        u.node[i]['freq'] = v
    graphdata = json_graph.node_link_data(u)
    graphdata['links'] = [
        {
            'source': graphdata['nodes'][link['source']]['id'],
            'target': graphdata['nodes'][link['target']]['id']
        }
        for link in graphdata['links']]
    return graphdata

def count_ngrams(corpus, q):
    nkey = q['qSettings']['nkey']
    trigram = q['qSettings']['trigram']
    ncorpus = normalise_text(corpus,q['qSettings'])
    words = getwords(ncorpus,q)

    word_fd = FreqDist(words)
    top_keywords = [(w, word_fd[w]) for w in sorted(word_fd, key=word_fd.get, reverse=True)]

    bigrams = getngrams(ncorpus,q,2)
    top_bigrams,bigram_fd,warning = findtopbigrams(bigrams,word_fd,q['qSettings'])
    top_ngrams = top_bigrams

    if trigram == "Trigram":
        trigrams = getngrams(ncorpus,q,3)
        top_trigrams,warning = findtoptrigrams(trigrams,word_fd,bigram_fd,q['qSettings'])
        top_ngrams = top_trigrams

    return top_keywords[:nkey],top_ngrams,warning


def findtopbigrams(bigrams,word_fd,settings):
    nkey = settings['nkey']
    measure = settings['measure']

    bigram_measures = BigramAssocMeasures()
    bigram_fd = FreqDist(bigrams)
    finder = BigramCollocationFinder(word_fd, bigram_fd)

    warning = ""

    if measure == "LR":
        try:
            top_bigrams = finder.nbest(bigram_measures.likelihood_ratio, nkey)
        except:
            warning = "Problem with LR measure. Default to simple frequency (RAW setting)"
            print(warning)
            top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)
    elif measure == "PMI":
        try:
            top_bigrams = finder.nbest(bigram_measures.pmi, nkey)
        except:
            warning = "Problem with PMI measure. Default to simple frequency (RAW setting)"
            print(warning)
            top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)
    elif measure == "CHISQ":
        try:
            top_bigrams = finder.nbest(bigram_measures.chi_sq, nkey)
        except:
            warning = "Problem with CHISQ measure. Default to simple frequency (RAW setting)"
            print(warning)
            top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)
    elif measure == "STUDT":
        try:
            top_bigrams = finder.nbest(bigram_measures.student_t, nkey)
        except:
            warning = "Problem with STUDT measure. Default to simple frequency (RAW setting)"
            print(warning)
            top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)
    else:
        top_bigrams = finder.nbest(bigram_measures.raw_freq, nkey)


    #score bigrams using LR or similar measure but more helpful to end user to see raw counts + explain measure used in tool tip
    top_bg_with_count = sorted([(bg, count) for (bg, count) in finder.ngram_fd.items() if bg in top_bigrams], key=lambda bgcount:-bgcount[1])
    top_bigrams = [(bg, count) for (bg, count) in top_bg_with_count if count > 1 and bg[0]!=bg[1]]
    return top_bigrams, bigram_fd, warning

def findtoptrigrams(trigrams,word_fd,bigram_fd,settings):
    nkey = settings['nkey']
    measure = settings['measure']

    trigram_measures = TrigramAssocMeasures()
    trigram_fd = FreqDist(trigrams)

    wild = [(t[0],t[2]) for t in trigrams]
    wild_fd = FreqDist(wild)
    finder = TrigramCollocationFinder(word_fd, bigram_fd, wild_fd, trigram_fd)

    warning = ""

    if measure == "LR":
        try:
            top_trigrams = finder.nbest(trigram_measures.likelihood_ratio, nkey)
        except:
            warning = "Problem with LR measure. Default to simple frequency (RAW setting)"
            print(warning)
            top_trigrams = finder.nbest(trigram_measures.raw_freq, nkey)
    elif measure == "PMI":
        try:
            top_trigrams = finder.nbest(trigram_measures.pmi, nkey)
        except:
            warning = "Problem with PMI measure. Default to simple frequency (RAW setting)"
            print(warning)
            top_trigrams = finder.nbest(trigram_measures.raw_freq, nkey)
    elif measure == "CHISQ":
        try:
            top_trigrams = finder.nbest(trigram_measures.chi_sq, nkey)
        except:
            warning = "Problem with CHISQ measure. Default to simple frequency (RAW setting)"
            print(warning)
            top_trigrams = finder.nbest(trigram_measures.raw_freq, nkey)
    elif measure == "STUDT":
        try:
            top_trigrams = finder.nbest(trigram_measures.student_t, nkey)
        except:
            warning = "Problem with STUDT measure. Default to simple frequency (RAW setting)"
            print(warning)
            top_trigrams = finder.nbest(trigram_measures.raw_freq, nkey)
    else:
        top_trigrams = finder.nbest(trigram_measures.raw_freq, nkey)

    #score trigrams using LR or similar measure but more helpful to end user to see raw counts + explain measure used in tool tip
    top_tg_with_count = sorted([(tg, count) for (tg, count) in finder.ngram_fd.items() if tg in top_trigrams], key=lambda tgcount:-tgcount[1])
    top_trigrams = [(tg, count) for (tg, count) in top_tg_with_count if count > 1 and tg[0]!=tg[1]]
    return top_trigrams, warning


def content_words_in_response(tokenised_response):
    content_words = 0
    tokenised_response = [t for tok in tokenised_response for t in tok] #flatten - ignore sentence breaks in list
    for t in tokenised_response:
        if t[1][:3] in ('ADJ','NOU','ADV','VER'): # count verbs, adjectives, adverbs and nouns as content words
            content_words +=1
    return content_words

def analysecorpus(corpus, q):
    return count_ngrams(corpus,q)

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

def ttr(n_unique_words,n_words):
    return 0 if n_words == 0 else n_unique_words/n_words

def lexical_density(content_words,total_words):
    return 0 if total_words == 0 else content_words/total_words

def smog(n_polysyllable_words,n_sents):
    return (1.0430 * sqrt(30 * n_polysyllable_words / n_sents)) + 3.1291

def gunning_fog(n_words,n_sents,n_polysyllable_words):
    return 0.4 * ((n_words / n_sents) + (100 * n_polysyllable_words / n_words))

def flesch_ease(n_words,n_syllables,n_sents):
    return (-84.6 * n_syllables / n_words) - (1.015 * n_words / n_sents) + 206.835

def flesch_kincaid(n_words,n_syllables,n_sents):
    return (11.8 * n_syllables / n_words) + (0.39 * n_words / n_sents) - 15.59