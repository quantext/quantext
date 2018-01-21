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
import collections
import pandas as pd
import textacy

import nlputility as nlp

from tft import app

#Following paths for standalone tests. Replace with ... in app
#DATAPATH = '../../SampleData/'
#CORPUSPATH = './TestData/'
#

class SAQData:
    """Instance of Short-Answer Question Data required for each data file
    Ultimately, want to be able to compare and contrast by student, by course etc. For now, embryonic prototype => focus on SAQ and TeachingData"""

    def __init__(self):
        print('init SAQ Data ...')

    def getdata_sample(self, datafname):
        #file upload here - should deal with filetypes etc
        #for now just deal with one type and hang the consequences...
        data = pd.ExcelFile(os.path.join(app.config['FILES_FOLDER'],datafname))
        return data

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

        qdata = collections.OrderedDict(zip(map(str,qdata[0]), qdata[1]))
        #populate response dictionary question id is the key - responses should be on separate sheets named by question id
        for key in qdata:
            if key != 'qIndex':
                rdata.update({key:data.parse(key, header = None, )})
                rdata[key].rename(columns = {0:"Response",1:"ID"}, inplace = True)
                rdata[key].dropna(inplace = True) #drop any na values

        return qdata, rdata

    def getandparsedata_sample(self, datafname):
        data = self.getdata_sample(datafname)
        return self.parsedata(data)

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
            qcorpus.add_text(response, metadata={"ID":str(ids), "categories":[], "notes":""})

        return qcorpus

    def loadcorpus(self, path, q):
        qcorpus = textacy.Corpus.load(path,name=str(q.qNum),compression='gzip')
        return qcorpus

    def get_worksheet(self, docs, reference_answer, search=None):
        #produce summary data for specified question
        s = textacy.Doc(reference_answer,lang='en')
        worksheetlist = []
        for doc in docs:
            newdict = {'StudentID':doc.metadata["ID"]}
            newdict.update({'Full_response':doc.text})
            if search is None:
                newdict.update({'Response':doc.text})
            else:
                newdict.update({'Response':list(textacy.text_utils.KWIC(doc.text,keyword=search,print_only=False))})

            newdict.update({'Category':','.join(doc.metadata["categories"])})
            newdict.update({'Notes':"" if "notes" not in doc.metadata else doc.metadata["notes"]})

            words = nlp.get_words(doc)
            n_words = len(words)
            n_unique_words = len({word.lower for word in words})
            n_polysyllable_words = nlp.count_syllables(words)[1]
            n_sents = doc.n_sents if n_words > 0 else 0

            newdict.update({'Words':n_words})
            newdict.update({'Sentences':n_sents})
            content_words = nlp.content_words_in_response(doc.pos_tagged_text)
            if n_words !=0:
                newdict.update({'TTR':nlp.ttr(n_unique_words,n_words)})
                newdict.update({'LD':nlp.lexical_density(content_words,n_words)})
            else:
                newdict.update({'TTR':0}) #this was None, now 0
                newdict.update({'LD':0}) #this was None, now 0
            if n_sents != 0:
                newdict.update({'SMOG':nlp.smog(n_polysyllable_words,n_sents)}) #Simple Measure of Gobbledygook - best for 30+ word responses
            else:
                newdict.update({'SMOG':0}) #this was None, now 0

            newdict.update({'Similarity':textacy.similarity.word2vec(s,doc)})
            worksheetlist.append(newdict)

        #convert list of dictionaries to pandas dataframe
        worksheet = pd.DataFrame.from_dict(worksheetlist)
        worksheet.sort_values(by = 'Words', ascending=True)
        worksheet = worksheet.round({'LD':2,'TTR':2,'SMOG':2,'Similarity':2})
        return worksheet

    def get_means(self, worksheet):
        means = {}
        means["Words"] =  "{0:.0f}".format(worksheet['Words'].mean())
        means["Sentences"] = "{0:.0f}".format(worksheet['Sentences'].mean())
        means["LD"] = "{0:.2f}".format(worksheet['LD'].mean())
        means["TTR"] = "{0:.2f}".format(worksheet['TTR'].mean())
        means["SMOG"] = "{0:.2f}".format(worksheet['SMOG'].mean())
        return means

    def analyseqcorpus(self, qcorpus, q):
        #returns top_keywords,top_ngrams,warning
        return nlp.analysecorpus(qcorpus,q)

    def getcollocgraph(self,qcorpus,settings):
        #returns json formatted d3 collocation graphdata
        return nlp.buildcollocgraph(qcorpus,settings)

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

        #self.corpus_root = 'tft/tmp/uploads/' #for pythonanywhere server
        self.corpus_root = 'tmp/uploads/' #for local server
        self.corpus_root_sample = 'tft/static/files/' #for local server

    def createtcorpus_sample(self, filename, refId):
        # just reading one text for now - use tcorpus.add_texts() when multifile upload implemented
        tcorpus = textacy.Corpus("en")
        text_to_add = textacy.fileio.read.read_file(self.corpus_root_sample + filename)
        tcorpus.add_text(text_to_add)

        origpath = os.path.join(app.config['CORPUS_FOLDER'],str(refId))
        if not os.path.exists(origpath):
            os.makedirs(origpath)

        fname = filename.rsplit('.', 1)[0]
        path = os.path.join(origpath,fname)
        if not os.path.exists(path):
            os.makedirs(path)
        tcorpus.save(path,name=fname,compression="gzip")
        return tcorpus

    def createtcorpus(self, filename, refId):
        # just reading one text for now - use tcorpus.add_texts() when multifile upload implemented
        tcorpus = textacy.Corpus("en")
        text_to_add = textacy.fileio.read.read_file(self.corpus_root + filename)
        tcorpus.add_text(text_to_add)

        origpath = os.path.join(app.config['CORPUS_FOLDER'],str(refId))
        if not os.path.exists(origpath):
            os.makedirs(origpath)

        fname = filename.rsplit('.', 1)[0]
        path = os.path.join(origpath,fname)
        if not os.path.exists(path):
            os.makedirs(path)
        tcorpus.save(path,name=fname,compression="gzip")
        return tcorpus

    def loadcorpus(self, path, filename):
        tcorpus = textacy.Corpus.load(path,name=filename,compression='gzip')
        return tcorpus

    def summarisetcorpus(self, tcorpus):
        n_sents = 0
        words = []
        pos_tagged_text = []
        for d in tcorpus.docs:
            new_words = list(textacy.extract.words(d, filter_punct=True, filter_stops=False, filter_nums=False))
            words.extend(new_words)
            n_sents += d.n_sents
            pos_tagged_text.extend(d.pos_tagged_text)

        n_syllables, n_polysyllable_words = nlp.count_syllables(words)
        totalsents = n_sents
        totalwords = len(words)
        unique_words = len({word.lower for word in words})
        totalchars = sum(len(word) for word in words)
        content_words = nlp.content_words_in_response(pos_tagged_text)

        lexicaldiversity = "{0:.2f}".format(nlp.ttr(unique_words,totalwords))
        lexicaldensity = "{0:.2f}".format(nlp.lexical_density(content_words,totalwords))
        smog = "{0:.2f}".format(nlp.smog(n_polysyllable_words,n_sents))
        gunning = "{0:.2f}".format(nlp.gunning_fog(totalwords,n_sents,n_polysyllable_words))
        flesch_ease = "{0:.2f}".format(nlp.flesch_ease(totalwords,n_syllables,n_sents))
        fk = "{0:.2f}".format(nlp.flesch_kincaid(totalwords,n_syllables,n_sents))

        return totalchars,totalwords,unique_words,totalsents,lexicaldiversity,lexicaldensity,smog,gunning,flesch_ease,fk

    def analysetcorpus(self, tcorpus, settings):
        key_count,bg_count,warning = nlp.analysecorpus(tcorpus, {"qSettings":settings,"blist":[],"wlist":[]})
        return key_count,bg_count,warning

    def kwicbyt(self, tcorpus, keyword):
        return nlp.kwic(tcorpus, keyword)

    def bigrambyt(self, tcorpus, keyword_phrase):
        return nlp.kwic(tcorpus, keyword_phrase)
