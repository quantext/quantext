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
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from nltk import agreement
from nltk.metrics import masi_distance

from tft import app

#Following paths for standalone tests. Replace with ... in app
#DATAPATH = '../../SampleData/'
#CORPUSPATH = './TestData/'

def getdata(datafname, folder):
    filepath = os.path.join(app.config[folder],datafname)
    name, ext = os.path.splitext(datafname)
    if ext == ".xlsx" or ext == ".xls":
        data = pd.read_excel(filepath,encoding='utf-8')
    elif ext == ".csv":
        with open(filepath) as f:
            data = pd.read_csv(filepath,encoding=f.encoding)

    return data

def read_pickle(pickle_path):
    data = pd.read_pickle(pickle_path)
    return data

def new_dataframe():
    return pd.DataFrame()

def make_copy(df):
    return pd.DataFrame(index=df.index)

def merge(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)

def get_totals(total_statistics):

    n_chars = sum([x['Characters'] for k,v in total_statistics.items() for x in v])
    n_words = sum([x['Words'] for k,v in total_statistics.items() for x in v])
    n_sents = sum([x['Sentences'] for k,v in total_statistics.items() for x in v])

    totals = {'Words':n_words,'Sentences':n_sents,'Characters':n_chars}
    return totals

def get_mean(responses):
    mean = np.mean(responses)
    return mean

def get_means(responses):
    word_mean = np.mean([r["Words"] for r in responses])
    sentence_mean = np.mean([r["Sentences"] for r in responses])
    means = {"Words":"{0:.0f}".format(word_mean)}
    means.update({"Sentences":"{0:.0f}".format(sentence_mean)})

    return means

def get_indices_means(indices):
    ld_mean = np.mean(np.array([i["LD"] for i in indices]).astype(np.float))
    ttr_mean = np.mean(np.array([i["TTR"] for i in indices]).astype(np.float))
    smog_mean = np.mean(np.array([i["SMOG"] for i in indices]).astype(np.float))
    gunning_mean = np.mean(np.array([i["Gunning"] for i in indices]).astype(np.float))
    flesch_mean = np.mean(np.array([i["Flesch"] for i in indices]).astype(np.float))
    fk_mean = np.mean(np.array([i["FK"] for i in indices]).astype(np.float))

    #new means
    mtld_mean = np.mean(np.array([i["MTLD"] for i in indices]).astype(np.float))
    hdd_mean = np.mean(np.array([i["HDD"] for i in indices]).astype(np.float))

    means = {"LD":"{0:.2f}".format(ld_mean)}
    means.update({"TTR":"{0:.2f}".format(ttr_mean)})
    means.update({"SMOG":"{0:.2f}".format(smog_mean)})
    means.update({"Gunning":"{0:.2f}".format(gunning_mean)})
    means.update({"Flesch":"{0:.2f}".format(flesch_mean)})
    means.update({"FK":"{0:.2f}".format(fk_mean)})

    means.update({"MTLD":"{0:.2f}".format(mtld_mean)})
    means.update({"HDD":"{0:.2f}".format(hdd_mean)})
    return means