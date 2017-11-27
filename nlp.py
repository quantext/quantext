"""
Created on Thu Dec  8 13:13:30 2016

@author: Jenny
"""

import os
import pandas as pd
import textacy

import nlputility as nlp

from tft import app

#SAVEPATH = './corpora/'        

class SAQData:
    """Instance of Short-Answer Question Data required for each data file
    Ultimately, want to be able to compare and contrast by student, by course etc. For now, embryonic prototype => focus on SAQ and TeachingData"""

    def __init__(self):
        print('init SAQ Data ...')

    def getdata(self, datafname):
        #file upload here - should deal with filetypes etc
        #for now just deal with one type and hang the consequences...
        data = pd.ExcelFile(os.path.join(app.config['UPLOAD_FOLDER'],datafname))
        return data

    def parsedata(self, data):
        #identify questions and responses
        #clean data - should also do the unicode sandwich thing...
        
        print('parsing data file ...')
        rdata = {} # initialise dict for responses

        qdata = data.parse('qIndex', header=None) #Must be one sheet in excel file called qIndex with two columns question id and question
        qdata = dict(zip(qdata[0], qdata[1]))
        qdata = dict((str(k), v) for k, v in qdata.items())
        #populate response dictionary question id is the key - responses should be on separate sheets named by question id
        for key in qdata:
            if key != 'qIndex':
                rdata.update({key:data.parse(key, header = None, )})
                rdata[key].rename(columns = {0:"Response",1:"ID"}, inplace = True)
                rdata[key].dropna(inplace = True) #drop any na values

        return qdata, rdata

    def getandparsedata(self, datafname):
        data = self.getdata(datafname)
        return self.parsedata(data)
        
    def createqcorpus(self, q):
        #Want to do this once only for each question since time intensive - when have db - should serialise etc..
        responses = q["Response"]  
        ids = q["ID"]              
        
        data = zip(responses,ids)
        qcorpus = textacy.Corpus("en") #initialise corpus for question
        
        for response,ids in data:
            qcorpus.add_text(response, metadata=str(ids))
            
        return qcorpus

    def loadcorpus(self, path, q):
        qcorpus = textacy.Corpus.load(path,name=str(q.qNum),compression='gzip')
        nlp.set_stoplist(qcorpus,q.qSettings)
        return qcorpus
        
    def createworksheet(self, qcorpus,modelanswer):
        #produce summary data for specified question
        s = textacy.Doc(modelanswer)
        worksheetlist = []
        for doc in qcorpus.docs:
            newdict = {}
            newdict.update({'StudentID':doc.metadata})
            newdict.update({'Response':doc.text})
            
            words = nlp.get_words(doc)
            n_words = len(words)
            n_unique_words = len({word.lower for word in words})
            n_polysyllable_words = nlp.count_syllables(words)[1]
            
            if n_words > 0:
                n_sents = doc.n_sents
            else:
                n_sents = 0
                
            newdict.update({'n_Words':n_words})
            newdict.update({'n_Sents':n_sents})
            if n_words !=0:
                newdict.update({'TTR':n_unique_words/n_words})
                newdict.update({'Lex_Density':nlp.content_words_in_response(doc.pos_tagged_text)/n_words})
            else:
                newdict.update({'TTR':0}) #this was None, now 0
                newdict.update({'Lex_Density':0}) #this was None, now 0
            if n_sents != 0:                
                newdict.update({'Smog_index':textacy.text_stats.smog_index(n_polysyllable_words,n_sents)}) #Simple Measure of Gobbledygook - best for 30+ word responses
            else:    
                newdict.update({'Smog_index':0}) #this was None, now 0

            newdict.update({'Similarity':textacy.similarity.word2vec(s,doc)})
            worksheetlist.append(newdict)

        #convert list of dictionaries to pandas dataframe
        sumqresps = pd.DataFrame.from_dict(worksheetlist)
        sumqresps.sort_values(by = 'n_Words', ascending=True)

        totalwords =  "{0:.0f}".format(sumqresps['n_Words'].mean())
        totalsents = "{0:.0f}".format(sumqresps['n_Sents'].mean())
        lexicaldensity = "{0:.2f}".format(sumqresps['Lex_Density'].mean())
        lexicaldiversity = "{0:.2f}".format(sumqresps['TTR'].mean())
        smog = "{0:.2f}".format(sumqresps['Smog_index'].mean())
        sumqresps = sumqresps.round({'Lex_Density':2,'TTR':2,'Smog_index':2,'Similarity':2})

        return sumqresps,totalwords,totalsents,lexicaldiversity,lexicaldensity,smog

    def analyseqcorpus(self, qcorpus, settings):
        key_count,bg_count,warning = nlp.analysecorpus(qcorpus,settings)
        return key_count,bg_count,warning

    def kwicbyq(self,qcorpus,keyword):
        return nlp.kwic(qcorpus,keyword)
    
    def bigrambyq(self,qcorpus,keyword_phrase):
        return nlp.kwic(qcorpus,keyword_phrase)

###########################################################################################################################################################

###########################################################################################################################################################        
    
class TeachingData:
    """Instance of Teaching Data can handle multiple files thro' textacy.filio.read.read() - just need to implement UI to support multi-upload.
    """
    
    def __init__(self):
    
        print('init teaching Data ...')

        self.corpus_root = 'tft/tmp/uploads/' #for pythonanywhere server
#        self.corpus_root = 'tmp/uploads/' #for local server

    def createtcorpus(self, filename, analysisId):
        # just reading one text for now - use tcorpus.add_texts() when multifile upload implemented
        tcorpus = textacy.Corpus("en")
        text_to_add = textacy.fileio.read.read_file(self.corpus_root + filename)
        tcorpus.add_text(text_to_add)

        origpath = os.path.join(app.config['CORPUS_FOLDER'],analysisId)
        if not os.path.exists(origpath):
            os.makedirs(origpath)

        fname = filename.rsplit('.', 1)[0]
        path = os.path.join(origpath,fname)
        if not os.path.exists(path):
            os.makedirs(path)
        tcorpus.save(path,name=fname,compression="gzip")
        return tcorpus

    def loadcorpus(self, path, filename, settings):
        tcorpus = textacy.Corpus.load(path,name=filename,compression='gzip')
        nlp.set_stoplist(tcorpus,settings)
        return tcorpus
        
    def summarisetcorpus(self, tcorpus):
        for d in tcorpus.docs:
            stats = textacy.text_stats.readability_stats(d)
            ld = nlp.content_words_in_response(d.pos_tagged_text)/stats['n_words']
        
        totalwords =  stats['n_words']
        totalsents = stats['n_sents']
        lexicaldensity = "{0:.2f}".format(ld)
        lexicaldiversity = "{0:.2f}".format(stats['n_unique_words']/stats['n_words'])
        smog = "{0:.2f}".format(stats['smog_index'])

        return totalwords,totalsents,lexicaldiversity,lexicaldensity,smog

    def analysetcorpus(self, tcorpus, settings):
        key_count,bg_count,warning = nlp.analysecorpus(tcorpus, settings)
        return key_count,bg_count,warning

    def kwicbyt(self, tcorpus, keyword):
        return nlp.kwic(tcorpus, keyword)
    
    def bigrambyt(self, tcorpus, keyword_phrase):
        return nlp.kwic(tcorpus, keyword_phrase)
    
