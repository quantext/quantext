"""
Created on Tues Jan  3 14:37:30 2017

@author: Jenny

Acknowledgements:
# Perkins' antonym replacer - uses wordnet for antonym lookup - avoids negation!
# Perkins' regexp replacer for normalising contractions

"""
from nltk import ConcordanceIndex
from nltk.corpus import wordnet
from nltk.metrics import edit_distance
import pandas as pd
import re
import enchant
from gensim import corpora, models, similarities
from six import iteritems

# replacement patterns should be user configurable
REPLACEMENT_PATTERNS = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'i\'d', 'i had'),
    (r'let\'s', 'let us'),
    (r'don\'t', 'do not'),
    (r'dont', 'do not'),
    (r'didn\'t', 'did not'),
    (r'it\'s', 'it is'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would')
    ]


class Spellcheck(object):
    """
    Allows specifying dictionary from Enchant module, custom
    keyword list and edit distances. Provides a 'correct' function to return best word 
    given input word. Provides for custom wordlist to be combined with standard dictionary.
    """
    
    def __init__(self, dict = "en_NZ", custom = None, maxdist = 2, maxdistkword = 3):
        
        #provide for adding custom dictionary - one word per line
        self.kwords = []
        if custom:
            for line in pd.read_csv(open(custom)):
                kword = ''.join(line)
                self.kwords.append(kword)
                
        self.maxdistkword = maxdistkword # higher value makes it less likely to correct words found in kword list
        self.maxdist = maxdist # higher value makes it more likely to correct words
 
        # initialise with default dictionary
        self.d = enchant.DictWithPWL(dict)  
    
    def correct(self, word):       
        
        if self.d.check(word):
            return(word)
        
        suggestedwords = self.d.suggest(word)
        print (word, suggestedwords)
 
        for sword in suggestedwords:
            if (sword in self.kwords) and (edit_distance(word, sword) <= self.maxdistkword):
                word = sword
                return(word)
        if suggestedwords and edit_distance(word, suggestedwords[0]) <= self.maxdist:
            return(suggestedwords[0])
        else:
            return(word)
            
            

class NgramConcordance(ConcordanceIndex):
    """
    Extends concordanceIndex to produce a printable concordance for unigram or bigram within 
    a specified context window. Still to do - add function to handle trigrams and 
    create helper function for creating output list.
    """
    def __init__(self, tokens, key=lambda x:x):
        ConcordanceIndex.__init__(self, tokens, key=lambda x:x)
    
    def print_bg_concordance(self, bg, width = 100, lines = 40, complete=True):
        w1,w2 = bg
        half_width = (width - len(w1) - len(w2) - 2) // 2
        context = width // 4 # approx number of words of context

        offsetw1 = self.offsets(w1)
        offsetw2 = self.offsets(w2)

        #iterate thro' list of offsetw1 and check if offsetw1 + 1 is in offsetw2 
        #If it is, then we have our bigram so store offsetw1 in list of bg offsets
        
        offsets = []
        output = []
        
        for offset in offsetw1:
            if offset+1 in offsetw2:
                offsets.append(offset)
          
        # deal with situations where bigrams sometimes hyphenated   
            if offset+1 in self.offsets("-") and offset+2 in offsetw2:
                offsets.append(offset)
        
        # build list of bgic to return for output
        if offsets:
            if not complete:
                lines = min(lines, len(offsets))
            else:
                lines = len(offsets)
            print("Returning %s of %s matches:" % (lines, len(offsets)))
            for i in offsets:
                if lines <= 0:
                    break
                left = (' ' * half_width +' '.join(self._tokens[i-context:i]))
                right = ' '.join(self._tokens[i+1:i+context])
                left = left[-half_width:]
                right = right[:half_width]
                output.append(left + ' ' + self._tokens[i] + ' ' + right)
                lines -= 1
            
        else:
            print("No matches")
        
        return output
        
    def print_kw_concordance(self, kword, width = 100, lines = 40, complete=True):

        half_width = (width - len(kword) - 1) // 2
        context = width // 4 # approx number of words of context

        offsets = self.offsets(kword)
        output = []
        
        # Just build list of kwic to return for output
        
        if offsets:
            if not complete:
                lines = min(lines, len(offsets))
            else:
                lines = len(offsets)
            print("Returning %s of %s matches:" % (lines, len(offsets)))
            for i in offsets:
                if lines <= 0:
                    break
                left = (' ' * half_width +' '.join(self._tokens[i-context:i]))
                right = ' '.join(self._tokens[i+1:i+context])
                left = left[-half_width:]
                right = right[:half_width]
                output.append(left + ' ' + self._tokens[i] + ' ' + right)
                lines -= 1
            
        else:
            print("No matches")
        
        return output
    
class SimilarResponses(object):
    
    def __init__(self, corpus, data, modeltype = 'LSI'):
        
        dictionary = corpora.Dictionary(corpus)
        once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
        
        dictionary.filter_tokens(once_ids)
        dictionary.compactify() #remove gaps in id sequence after words removed
        
        self.dictionary = dictionary

        self.corpus = [[word.lower() for word in resp] for resp in corpus]
        self.data = data
        
        self.modeltype = modeltype  #not used yet - only using LSI      
        self.model = self.buildmodel()
        
    def buildmodel(self, corpus = None, num_topics = 2):
        
        #allows option of using corpus different to that from which dictionary created
        if not corpus:
            corpus = self.corpus 
            
        corpus = [self.dictionary.doc2bow(item) for item in corpus]
        
        #LSI corpus transformation
        tfidf = models.TfidfModel(corpus, id2word = self.dictionary)
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, num_topics, id2word = self.dictionary)
        corpus_lsi = lsi[corpus_tfidf]
        
        #print words in order of association with topics - add model switch later. For now just use LSI
        print("LSI Topics")
        print(lsi.print_topics())

        #LDA corpus transformation
        #lda = models.LdaModel(corpus, id2word = self.dictionary, num_topics=2)
        #corpus_lda = lda[corpus]

        #print words in order of association with topics
        #print "LDA Topics"
        #print(lda.print_topics())

        #HDP corpus transformation
        #hdp = models.HdpModel(corpus, id2word = self.dictionary)
        #corpus_hdp = hdp[corpus]

        #print words in order of association with topics
        #print "HDP Topics"
        #print(hdp.print_topics())

        return corpus_lsi,lsi,tfidf,
        
    def findsimilarity(self, query):
        
        vec_bow = self.dictionary.doc2bow(query.lower().split())

        #LSI vector transformation
        vec_tfidf = self.model[2][vec_bow] #tfidf
        vec_lsi = self.model[1][vec_tfidf] #lsi
        print
        print(vec_lsi, query)


        #select model and vector transform and print docs by model - just lsi for now
        model = self.model[0] #corpus_lsi
        q_vec = vec_lsi
        #docs = [line for line in open(data)]


        #generate similarity matrix and visualise results
        index = similarities.MatrixSimilarity(model)
        sims = index[q_vec]
        simsidx = dict(enumerate(zip(sims,self.data)))

        #sort results by similarity - most to least
        simsorted = sorted([(k,v) for (k,v) in simsidx.items()], key=lambda(k,v): -v[0])
        
        return simsorted
        
        
# Perkins Replacer classes
# utility class for replacing contractions
class RegexpReplacer(object):
    def __init__(self, patterns = REPLACEMENT_PATTERNS):
        self.patterns = [(re.compile(regex), repl) for (regex,repl) in patterns]
    
    def replace(self,text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s
        
#utility class to replace a word for Antonym and Synonym replacers - not used yet. Should be user configurable and useful for model building
class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map
    def replace(self, word):
        return self.word_map.get(word, word)


#find unambiguous antonym from wordnet and replace negation
class AntonymReplacer(object):
    def replace(self, word, pos=None):
        antonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas:
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name)
        if len(antonyms) == 1:
            return antonyms.pop()
        else:
            return None
        
    def replace_negations(self, sent):
        i,l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i+1 < l:
                ant = self.replace(sent[i+1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words

#read in custom word mapping from text file
class CsvWordReplacer(WordReplacer):
    def __init__(self,fname):
        word_map = {}
        for line in pd.read_csv(open(fname)):
            word,rep = line
            word_map[word] = rep
        super(CsvWordReplacer, self).__init__(word_map)
        
#Wrapper classes

class AntonymWordReplacer(CsvWordReplacer, AntonymReplacer):
    pass

class SynonymWordReplacer(CsvWordReplacer):
    pass



