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
import pandas as pd
import textacy

import nlputility as nlp

from tft import app

#Following paths for standalone tests. Replace with ... in app
#DATAPATH = '../../SampleData/'
#CORPUSPATH = './TestData/'

def blank_corpus():
    return textacy.Corpus("en")

def getdata_sample(datafname):
    #file upload here - should deal with filetypes etc
    #for now just deal with one type and hang the consequences...
    data = pd.ExcelFile(os.path.join(app.config['FILES_FOLDER'],datafname),encoding='utf-8')
    return data

def getdata(datafname):
    #file upload here - should deal with filetypes etc
    #for now just deal with one type and hang the consequences...
    data = pd.ExcelFile(os.path.join(app.config['UPLOAD_FOLDER'],datafname),encoding='utf-8')
    return data

def parsedata(data):
    #identify questions and responses
    #clean data - should also do the unicode sandwich thing...

    print('parsing data file ...')
    rdata = {} # initialise dict for responses
    qdata = data.parse('qIndex', header=None) #Must be one sheet in excel file called qIndex with two columns question id and question, or 3 columns question id, question and reference answer
    rows, columns = qdata.shape

    if columns == 2:
        subset = qdata[[0,1]]
    else:
        subset = qdata[[0,1,2]]
    qdata = [tuple(x) for x in subset.values]
    #populate response dictionary question id is the key - responses should be on separate sheets named by question id
    for key,*v in qdata:
        if key != 'qIndex':
            key = str(key)
            d = data.parse(key, header = None,dtype='object')
            rdata.update({key:d})
            rdata[key].rename(columns = {0:"Response",1:"ID",2:"PostID",3:"ParentID"}, inplace = True)
            rdata[key] = rdata[key][rdata[key]["Response"].notnull()]

    return qdata, rdata

def getandparsedata_sample(datafname):
    data = getdata_sample(datafname)
    return parsedata(data)

def getandparsedata(datafname):
    data = getdata(datafname)
    return parsedata(data)

def createqcorpus(q):
    data = [tuple(x) for x in q.values]
    qcorpus = textacy.Corpus("en") #initialise corpus for question

    for r,*items in data:
        response = " ".join(r.split())
        if len(items) == 1:
            qcorpus.add_text(response, metadata={"ID":str(items[0]),"categories":[], "notes":""})
        elif len(items) == 2:
            qcorpus.add_text(response, metadata={"ID":str(items[0]),"PostID":str(items[1]),"categories":[], "notes":""})
        elif len(items) == 3:
            qcorpus.add_text(response, metadata={"ID":str(items[0]),"PostID":str(items[1]),"ParentID":str(items[2]),"categories":[], "notes":""})

    return qcorpus

def loadcorpus(path, q):
    qcorpus = textacy.Corpus.load(path,name=str(q.qNum),compression='gzip')
    return qcorpus

def get_worksheet(corpus, reference_answer, search=None):
    ref_doc = textacy.Doc(reference_answer,lang='en')
    worksheetlist = []

    for doc in corpus.docs:
        if "PostID" in doc.metadata and doc.metadata["PostID"] != "":
            newdict = {'StudentID':doc.metadata["PostID"]}
            newdict.update({'DT_RowId':doc.metadata["PostID"]})
        else:
            newdict = {'StudentID':doc.metadata["ID"]}
            newdict.update({'DT_RowId':doc.metadata["ID"]})

        newdict.update({'Full_response':doc.text})
        if search is None:
            newdict.update({'Response':doc.text})
        else:
            newdict.update({'Response':list(textacy.text_utils.KWIC(doc.text,keyword=search,print_only=False))})

        newdict.update({'Category':','.join(doc.metadata["categories"])})
        newdict.update({'Notes':"" if "notes" not in doc.metadata else doc.metadata["notes"]})
        if "PostID" in doc.metadata and doc.metadata["PostID"] != "":
            newdict.update({'PostID':doc.metadata["PostID"]})
            children = corpus.get(lambda d: d.metadata["ParentID"] == doc.metadata["PostID"])
            kids = []
            for c in children:
                kids.append({'text':c.text,'ID':c.metadata["PostID"]})
            newdict.update({'children':kids})
        else:
            newdict.update({'PostID':""})
            newdict.update({'children':""})

        newdict.update({'ParentID':"" if "ParentID" not in doc.metadata else doc.metadata["ParentID"]})

        words = nlp.get_words(doc)
        n_words = len(words)
        n_sents = doc.n_sents if n_words > 0 else 0
        newdict.update({'Words':n_words})
        newdict.update({'Sentences':n_sents})
        newdict.update({'Similarity':textacy.similarity.word2vec(ref_doc,doc)})
        worksheetlist.append(newdict)

    #convert list of dictionaries to pandas dataframe
    worksheet = pd.DataFrame.from_dict(worksheetlist)
    worksheet.sort_values(by = 'Words', ascending=True)
    worksheet = worksheet.round({'LD':2,'TTR':2,'SMOG':2,'Similarity':2})
    return worksheet

def get_means(corpus, reference_answer):
    ref_doc = textacy.Doc(reference_answer,lang='en')
    doclist = []
    for doc in corpus.docs:
        words = nlp.get_words(doc)
        n_words = len(words)
        n_unique_words = len({word.lower for word in words})
        n_syllables, n_polysyllable_words = nlp.count_syllables(words)
        n_sents = doc.n_sents if n_words > 0 else 0
        content_words = nlp.content_words_in_response(doc.pos_tagged_text)

        newdict = {'Words':n_words}
        newdict.update({'Sentences':n_sents})
        if n_words !=0:
            newdict.update({'TTR':nlp.ttr(n_unique_words,n_words)})
            newdict.update({'LD':nlp.lexical_density(content_words,n_words)})
        else:
            newdict.update({'TTR':0}) #this was None, now 0
            newdict.update({'LD':0}) #this was None, now 0
        if n_sents != 0:
            newdict.update({'SMOG':nlp.smog(n_polysyllable_words,n_sents)}) #Simple Measure of Gobbledygook - best for 30+ word responses
            newdict.update({'Gunning':nlp.gunning_fog(n_words,n_sents,n_polysyllable_words)})
            newdict.update({'Flesch':nlp.flesch_ease(n_words,n_syllables,n_sents)})
            newdict.update({'FK':nlp.flesch_kincaid(n_words,n_syllables,n_sents)})
        else:
            newdict.update({'SMOG':0}) #this was None, now 0
            newdict.update({'Gunning':0})
            newdict.update({'Flesch':0})
            newdict.update({'FK':0})

        newdict.update({'Similarity':textacy.similarity.word2vec(ref_doc,doc)})
        doclist.append(newdict)

    docframe = pd.DataFrame.from_dict(doclist)
    means = {"Words":"{0:.0f}".format(docframe['Words'].mean())}
    means.update({"Sentences":"{0:.0f}".format(docframe['Sentences'].mean())})
    means.update({"LD":"{0:.2f}".format(docframe['LD'].mean())})
    means.update({"TTR":"{0:.2f}".format(docframe['TTR'].mean())})
    means.update({"SMOG":"{0:.2f}".format(docframe['SMOG'].mean())})
    means.update({"Gunning":"{0:.2f}".format(docframe['Gunning'].mean())})
    means.update({"Flesch":"{0:.2f}".format(docframe['Flesch'].mean())})
    means.update({"FK":"{0:.2f}".format(docframe['FK'].mean())})
    return means, docframe

def analyseqcorpus(qcorpus, q):
    #returns top_keywords,top_ngrams,warning
    return nlp.analysecorpus(qcorpus,q)

def getcollocgraph(qcorpus,settings):
    #returns json formatted d3 collocation graphdata
    return nlp.buildcollocgraph(qcorpus,settings)

def kwicbyq(qcorpus,keyword):
    return nlp.kwic(qcorpus,keyword)

def bigrambyq(qcorpus,keyword_phrase):
    return nlp.kwic(qcorpus,keyword_phrase)

###########################################################################################################################################################

###########################################################################################################################################################

CORPUS_ROOT = 'tmp/uploads/'
CORPUS_ROOT_SAMPLE = 'tft/static/files/'

def createtcorpus_sample(filename, refId):
    # just reading one text for now - use tcorpus.add_texts() when multifile upload implemented
    tcorpus = textacy.Corpus("en")
    text_to_add = textacy.fileio.read.read_file(CORPUS_ROOT_SAMPLE + filename)
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

def createtcorpus(filename, refId):
    # just reading one text for now - use tcorpus.add_texts() when multifile upload implemented
    tcorpus = textacy.Corpus("en")
    text_to_add = textacy.fileio.read.read_file(CORPUS_ROOT + filename)
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

def loadtcorpus(path, filename):
    tcorpus = textacy.Corpus.load(path,name=filename,compression='gzip')
    return tcorpus

def summarisetcorpus(tcorpus):
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

def analysetcorpus(tcorpus, settings):
    key_count,bg_count,warning = nlp.analysecorpus(tcorpus, {"qSettings":settings,"blist":[],"wlist":[]})
    return key_count,bg_count,warning

def kwicbyt(tcorpus, keyword):
    return nlp.kwic(tcorpus, keyword)

def bigrambyt(tcorpus, keyword_phrase):
    return nlp.kwic(tcorpus, keyword_phrase)