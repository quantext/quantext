"""
Created on Thu Dec  8 13:13:30 2016

@author: Jenny
"""

import pandas as pd
import re
import os
import nlputility as nlp

from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


from tft import app

#nltk.data.path.append('./nltk_data/') #ensure can find data locally

#WHITELIST should be user configurable ...
WHITELIST = ['I', 'you', 'we', 'do', 'not', 'must', 'should', 'can', 'will']

class SAQData():
    """Instance of Short-Answer Question Data required for each data file
    Envisage separating out a class for loading data then building classes for other data types. 
    For example, EssayQData, DiscussionForumData, TeachingCorpusData etc. Ultimately, want to be 
    able to compare and contrast by student, by course etc. For now, embryonic prototype => focus on SAQ and incl loading data"""

    def __init__(self, datafname):

        print 'init SAQ Data ...'

        self.datafname = datafname
        self.pdata = self.parsedata(self.getdata(datafname))
        
        
        
    def getdata(self, datafname):
        #file upload here - should deal with filetypes etc
        #for now just deal with one type and hang the consequences...
        
        data = pd.ExcelFile(os.path.join(app.config['UPLOAD_FOLDER'],datafname))
        
        return data
        
        
    def parsedata(self, data):
        #identify questions and responses
        #clean data - should also do the unicode sandwich thing...
        
        print 'parsing data file ...'
        
        #Must be one sheet in excel file called qInex with two columns question id and question
        pdata = data.parse('qIndex', header=None)
        pdata = dict(zip(pdata[0], pdata[1]))
        #convert each value into first element of a list
        for (k,v) in pdata.items():
            pdata[k] = [v]
        #convert keys to strings just in case question id numeric or something other than string
        pdata = dict((str(k), v) for k, v in pdata.items())
        #append responses for each question to value list - responses should be on separate sheets named by question id
        for key in pdata:
            if key != 'qIndex':
                pdata[key].append(data.parse(key, header = None))
                pdata[key][1].rename(columns = {0:"Response"}, inplace = True)
                pdata[key][1].dropna(inplace = True) #drop any na values
        

            
        return pdata
        
    def summarisedatabyq(self, pdata, question):
        #produce summary data for specified question
        
        q = pdata[question][1]
        q["Length"] = q["Response"].map(lambda rsp: len(rsp)) 
        
        sumqresps = q[['Response', 'Length']].sort_values(by = 'Length', ascending=True)
        sumqstats = q["Length"].describe()
        
        return sumqresps, sumqstats
        
    def normalisedata(self, pdata, question):
        #tokenise, case conversion. Should get pulled out into own generic class at some point.
        q = pdata[question][1]
        r = nlp.RegexpReplacer()
        s = nlp.Spellcheck() #Include path to custom keyword file as a parameter, if there is one ...       
        
        #convert to ascii for string handling functions
        responses = [resp.encode('utf-8') for resp in q["Response"]]
        
        #deal to contractions - i.e. expand don't to do not etc
        responses = [r.replace(resp) for resp in responses]
        
        #tokenize
        tokenised_responses = [word_tokenize(response)for response in responses]


        #spellcheck corpus - words only, not tokens
        corpus = []
        for response in tokenised_responses:
            nresponse = []
            for token in response:
                if re.search(r'[A-z]', token):
                    token = s.correct(token) #only correct words or spell checker returns odd results...
                else:
                    token = token
                nresponse.append(token)
            corpus.append(nresponse)

        return corpus
        
        
    def analysedatabyq(self, corpus):
        #basic text analysis functions - viz: top keywords and top bigrams
    
        corpus = [t.lower() for response in corpus for t in response] #flatten and lower-case corpus for now. User should configure
        
        # generate most frequent words in corpus without stopwords
        stops = stopwords.words('english')
        stops = [word.encode('utf-8') for word in stops if word not in WHITELIST] #review stoplist
        
        #Find top keywords  - todo: parameters should be user configurable
        contentwords = [word for word in corpus if word not in stops and re.match(r'[A-z]+',word) and not \
                re.match(r'(\s+[A-z]+)|([A-z]+\s+)',word)]
        fdistcont = FreqDist(contentwords)
        top_keywords = fdistcont.most_common(10) #list top n words - should be user configurable
        
        #Find top bigrams # parameters can be customised
        finder = BigramCollocationFinder.from_words(contentwords) #default window size = 2 or 1L and 1R
        
        bigram_measures = BigramAssocMeasures()
        top_bigrams = finder.nbest(bigram_measures.likelihood_ratio, 10)
        top_bg_with_count = sorted([(bg, count) for (bg, count) in finder.ngram_fd.viewitems() if bg in top_bigrams], key=lambda(bg,count):-count)
        top_bigrams = [(bg, count) for (bg, count) in top_bg_with_count if count > 1]
        
        return top_keywords, top_bigrams
        
        
    def bgicbyq(self, corpus, bg, width=100, lines=40, complete = True):
        corpus = [t.lower() for response in corpus for t in response] #flatten and lower-case corpus for now. User should configure
        c = nlp.NgramConcordance(corpus)        
        bgic = c.print_bg_concordance(bg, width, lines)

        return bgic
            
    def kwicbyq(self, corpus, kword, width=100, lines=40, complete = True):
        corpus = [t.lower() for response in corpus for t in response] #flatten and lower-case corpus for now. User should configure
        c = nlp.NgramConcordance(corpus)        
        kwic = c.print_kw_concordance(kword, width, lines)
   
        return kwic
        
    def visualisedatabyq(self, corpus, pdata, question, query):
        #basic similarity of response visualisation function - rudimentary at this stage!
        
        q = pdata[question][1]
        responses = [resp.encode('utf-8') for resp in q["Response"]]
        
        sim = nlp.SimilarResponses(corpus, responses)
        similarlist = sim.findsimilarity(query)
        
        return similarlist
        
        
        