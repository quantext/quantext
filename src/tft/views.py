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


from flask import render_template, flash, url_for, redirect, request, jsonify, session
from tft import app, db
from werkzeug import secure_filename
#from .forms import LoginForm
from flask_login import login_user, logout_user, current_user
from oauth import OAuthSignIn
from oauth2client.client import OAuth2WebServerFlow
from .models import User, Analysis, Question
import nlp
import numpy

import json
import string
import random
import os
import re
import textract

s = nlp.SAQData()
t = nlp.TeachingData()

# initialise session vars - not sure this is at all secure!! Need to check :|
def default_settings():
    settings = {}

    settings['nkey'] = app.config['KEYWORDS']
    settings['kblack']=app.config['KEY_BLACKLIST']
    settings['bblack']=app.config['KEYPAIRS_BLACKLIST']
    settings['white']=app.config['ONLY_WHITELIST']
    settings['punct']=app.config['FILTER_PUNCT']
    settings['nums']=app.config['FILTER_NUMS']
    settings['measure']=app.config['BG_MEASURE']
    settings['blist'] = app.config['BLACKLIST']
    settings['wlist'] = app.config['WHITELIST']
    settings['ncontractions'] = app.config['NORM_CONTRACTIONS']

    settings['spell'] = app.config['SPELLCHECK']
    settings['stem'] = app.config['STEMMING']
    settings['negreplace'] = app.config['NEGATIVE_REPLACER']
    settings['synreplace'] = app.config['SYNONYM_REPLACER']
    settings['simalgorithm'] = app.config['SIM_ALGORITHM']
 
    return settings

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
    
# This function should run once when data file is loaded and store data in db with associated session id
# For now, just call this function whenever data required ...
# Globals don't work if multiple sessions - see Flask documentation ...    
def get_saq_data(filename):
    "Given a filename, parse file" 
    #s = nlp.SAQData()
    data = s.getdata(filename)
    s.qdata,s.rdata = s.parsedata(data)        
    return s
    
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

def analysis_and_question(id,question):
    an = Analysis.objects.get(id=id)
    q = [ques for ques in an.questions if ques.qNum == int(question)][0]
    return an,q

@app.route('/')
@app.route('/index')
@app.route('/index/<analysisId>')
def index(analysisId=None, filenamet=None):
    
    # initialise session vars from config file
    analyses = None
    if not current_user.is_anonymous:
        analyses = Analysis.objects(owner=current_user.id)

    filename = None
    if not analysisId is None:
        an = Analysis.objects.get(id=analysisId)
        filename = an.filename

    return render_template('index.html',
                           title = 'Text Analytics App',
                           analyses = analyses,
                           filename=filename,
                           analysisId = analysisId,
                           filenamet = filenamet)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/authorize/<provider>')
def oauth_authorize(provider):
    if not current_user.is_anonymous:
        return redirect(url_for('index'))
    if provider == 'twitter':
        oauth = OAuthSignIn.get_provider(provider)
        return oauth.authorize()
    else:
        return redirect(url_for('google_callback'))

@app.route('/google_callback')
def google_callback():
    url = request.url_root
    flow = OAuth2WebServerFlow(
        client_id=app.config['OAUTH_CREDENTIALS']['google']['web']['client_id'],
        client_secret=app.config['OAUTH_CREDENTIALS']['google']['web']['client_secret'],
        scope=['profile','email'],
        redirect_uri=url+'google_callback'
    )
    if 'code' not in request.args:
        auth_uri = flow.step1_get_authorize_url()
        return redirect(auth_uri)
    else:
        auth_code = request.args.get('code')
        credentials = flow.step2_exchange(auth_code).to_json()
        credentials = json.loads(credentials)
        id_token = credentials["id_token"]
        name = id_token["name"]
        email = id_token["email"]

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            try:
                user = User.objects.get(social_id=name)
                user["email"] = email
                user.save()
            except User.DoesNotExist:
                User(social_id=name, nickname=name, email=email).save()
                user = User.objects.get(social_id=name)
                create_demo_files(user)

        login_user(user, True)
        return redirect(url_for('index'))

@app.route('/callback/<provider>')
def oauth_callback(provider):
    if not current_user.is_anonymous:
        return redirect(url_for('index'))
    oauth = OAuthSignIn.get_provider(provider)
    social_id, username, email = oauth.callback()
    if social_id is None:
        flash('Authentication failed.')
        return redirect(url_for('index'))

    try:
        user = User.objects.get(social_id=social_id)
    except:
        User(social_id=social_id, nickname=username, email=email).save()
        user = User.objects.get(social_id=social_id)

    login_user(user, True)
    return redirect(url_for('index'))    

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500   

@app.route('/uploadq', methods = ['GET', 'POST'])
def upload_file_qs():
#This is bare minimum function with rudimenary error checking! 
#Need to:
#check for over-writing files, write filenames to db per user, decide how to handle file persistance??, 
#check for file size, decide file size limit? ...
#interface leaves much to be desired too - should be able to handle multi-uploads and then select files 
#for analysis etc, file preview would be handy...
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_fileq(f.filename):
            try:
                filename = fname_generator(secure_filename(f.filename))
                f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

                user = User.objects.get(id=current_user.id)
                an = Analysis(filename=filename, owner=user)
                an.save()

                origpath = os.path.join(app.config['CORPUS_FOLDER'],str(an.id))
                if not os.path.exists(origpath):
                    os.makedirs(origpath)

                # do the processing here on upload
                qdata, rdata = s.getandparsedata(filename)

                # save questions
                myQs = []
                idx = 1
                for num,text in qdata.items():
                    question = Question(qNum=idx,qTitle=num,qText=text)
                    qcorpus = s.createqcorpus(rdata[num])

                    settings = default_settings()
                    question.qSettings = settings

                    path = os.path.join(origpath,str(idx))
                    if not os.path.exists(path):
                        os.makedirs(path)

                    qcorpus.save(path,name=str(idx),compression="gzip")
                    idx = idx + 1
                    myQs.append(question)

                an.questions = myQs
                an.save()

                return redirect(url_for('index', analysisId=an.id))
            except Exception as e:
                flash('There is a problem uploading your file: ' + str(e))
            return redirect(url_for('index'))

        else:
            flash('Your file must have a .xls or .xlsx extension')

    return redirect(url_for('index'))
   
@app.route('/uploadt/<analysisId>', methods = ['GET', 'POST'])
def upload_file_teach(analysisId=None):
#This is bare minimum function to upload teaching materials
    print("analysis id")
    print(analysisId)
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_filet(f.filename):
            try:
                filenamet = fname_generator(secure_filename(f.filename))
                f.save(os.path.join(app.config['UPLOAD_FOLDER'],filenamet))
                if filenamet.rsplit('.', 1)[1] == 'pdf':     #txt file is just saved. Saved pdf file needs converted to txt
                    #if analysisId is not None:
                    an = Analysis.objects.get(id=analysisId)
                    p = os.path.join(app.config['UPLOAD_FOLDER'],filenamet)
                    f = pdf_to_text(p)
                    fname = filenamet.rsplit('.', 1)[0]
                    filenamet = fname + '.txt'
                    fout = open(os.path.join(app.config['UPLOAD_FOLDER'],filenamet),'w')
                    fout.write(f)
                    fout.close()
                    t.createtcorpus(filenamet,analysisId)
                    an.tcorpus = fname
                    an.save()

                #session['filenamet'] = filenamet
                return redirect(url_for('index', analysisId=analysisId, filenamet=filenamet))
            except Exception as e:
                flash('There is a problem uploading your file: ' + str(e))
            return redirect(url_for('index', analysisId=analysisId))
        else:
            flash('Your file must have a .pdf or .txt extension')

    return redirect(url_for('index', analysisId=analysisId))

 
@app.route('/analyse/', methods = ['GET'])
@app.route('/analyse/<analysisId>', methods = ['GET'])
@app.route('/analyse/<analysisId>/<question>', methods = ['GET'])
@app.route('/analyse/<analysisId>/<question>/<ngic>', methods = ['GET'])
def analyse(analysisId=None,question=None, ngic=None):
    if not question:
        question = request.form.get('selectq')

    an, q =  analysis_and_question(analysisId,question)
    modelanswer = q.modelAnswer
    if modelanswer is None:
        modelanswer = "Reference answer not set yet (click to set)"

    qcorpus = load_corpus(an,q)
    anaq = ana_q(qcorpus,q)
    t_kwords = None
    t_bgs = None
    meant = None

    if an.tcorpus:
        t_kwords,t_bgs,warning,meant = ana_t(an,q)
    wsdata,qcount,mean,chartdata,sents = sum_q(qcorpus,modelanswer)

    return render_template('analyse.html',
                           an=an,
                           analysisId=analysisId,
                           question=question,
                           ngic=ngic,
                           modelanswer=modelanswer,
                           anaq=anaq,
                           q=q,
                           wsdata=wsdata,
                           qcount=qcount,
                           mean=mean,
                           chartdata=chartdata,
                           sents=sents,
                           t_kwords=t_kwords,
                           t_bgs=t_bgs,
                           meant=meant)

@app.route('/visualise')   
@app.route('/visualise/<question>')
@app.route('/visualise/<question>/<ngic>')
def visualise(question=None, ngic=None):
     return render_template('visualise.html', question=question, ngic=ngic)

@app.route('/charts/<chartdata>/<charttype>')
def charts(chartdata=None, charttype=None):
    return render_template('charts.html', chartdata=chartdata, charttype=charttype)

@app.route('/about')
def about():
    return render_template('about.html', title = 'About Text Analytics App')
    
@app.route('/contact')
def contact():
    return render_template('contact.html', title = 'Contact InClass Team')
    
@app.route('/settings', methods = ['GET', 'POST'])
def settings(sets=None):
    id = request.form.get("id")
    question = request.form.get("question")
    an, q = analysis_and_question(id,question)
    key = str(request.form.get("key"))
    val = request.form.get("val")

    if key == 'blist' or key == 'wlist':
        val = json.loads(val)

    q.qSettings[key] = val
    q.save()
    qcorpus = load_corpus(an,q)
    anaq = ana_q(qcorpus,q)

    t_kwords = None
    t_bgs = None
    meant = None
    if an.tcorpus:
        t_kwords,t_bgs,warning,meant = ana_t(an,q)

    return render_template('keyword_render.html',
                           q=q,
                           question=question,
                           anaq=anaq,
                           t_kwords=t_kwords,
                           t_bgs=t_bgs,
                           meant=meant)
    
@app.route('/test')
def test():
    return render_template('test.html', title = 'About Text Analytics App')

@app.route('/_process_model_answer')
def process_model_answer():
    try:
        answer = request.args.get('modelanswer')
        id = request.args.get('id')
        question = request.args.get('question')

        an, q = analysis_and_question(id,question)
        q.modelAnswer = answer
        q.save()

        return jsonify(result=str(answer))
    except Exception as e:
        return str(e)

@app.template_filter('getqlist')
def get_qs(filename):
    an = Analysis.objects.get(filename=filename)
    return an.questions

def load_corpus(an,q):
    path = os.path.join(app.config['CORPUS_FOLDER'],str(an.id),str(q.qNum))
    qcorpus = s.loadcorpus(path,q)
    return qcorpus

def load_tcorpus(an,filename,settings):
    path = os.path.join(app.config['CORPUS_FOLDER'],str(an.id),filename)
    tcorpus = t.loadcorpus(path,filename,settings)
    return tcorpus

def sum_q(qcorpus,modelanswer):
    #summarise responses by length and extract key summary stats
    wsdata,meanwords,meansents,meanttr,meanld,meansmog = s.createworksheet(qcorpus,modelanswer)
    mean = {"meanwords":meanwords,"meansents":meansents,"meanttr":meanttr,"meanld":meanld,"meansmog":meansmog}
    qcount = len(wsdata)
    chartdata = get_chartdata(wsdata)
    hist, values = numpy.histogram(wsdata["n_Sents"].values)
    wsdata = zip(wsdata['StudentID'],wsdata['Response'],wsdata['n_Words'], wsdata['n_Sents'], wsdata['TTR'], wsdata['Lex_Density'], wsdata['Smog_index'], wsdata['Similarity'])

    return wsdata,qcount,mean,chartdata,list(hist)

def create_chart(wsdata, chartdata, ws_key):
    bins = len(set(wsdata[ws_key].values))
    if bins > 10:
        bins = 10
    hist, values = numpy.histogram(wsdata[ws_key].values, bins=bins)

    obj = {}
    for index,value in numpy.ndenumerate(hist):
        obj[values[index]] = value
    chartdata[ws_key] = obj
    return chartdata

def get_chartdata(wsdata):
    chartdata = create_chart(wsdata,{},"n_Words")
    chartdata = create_chart(wsdata,chartdata,"n_Sents")
    chartdata = create_chart(wsdata,chartdata,"TTR")
    chartdata = create_chart(wsdata,chartdata,"Lex_Density")
    chartdata = create_chart(wsdata,chartdata,"Smog_index")
    return chartdata

def ana_q(qcorpus,q):
    settings = q.qSettings
    top_kwords, top_bgs, warning = s.analyseqcorpus(qcorpus,settings)
    return {"top_kwords":top_kwords, "top_bgs":top_bgs}

def ana_t(an,q):
    filename = an.tcorpus
    settings = q.qSettings
    tcorpus = load_tcorpus(an, filename, settings)
    twords,tsents,tttr,tld,tsmog = t.summarisetcorpus(tcorpus)
    meant = {"twords":twords,"tsents":tsents,"tttr":tttr,"tld":tld,"tsmog":tsmog}
    qSettings = q.qSettings
    top_kwords,top_bgs,warning = t.analysetcorpus(tcorpus, qSettings)

    return top_kwords,top_bgs,warning,meant

@app.context_processor
def get_kwic():
 def _get_kwic(question, myword):
    
    #filename = session['filename']
    #s = get_saq_data(filename)
    #s.createqcorpus(s.rdata, question)
    #kwic = s.kwicbyq(myword)
    #kwic_complete = [[line[1] + ' ' + line[2] + ' ' + line[3]] for line in kwic] #required format for Wordtree visualisation
    
    return kwic
 return dict(get_kwic = _get_kwic)

@app.context_processor
def get_bgic():
 def _get_bgic(question, mybg):

    #filename = session['filename']
    #s = get_saq_data(filename)
    #s.createqcorpus(s.rdata, question)
    #bg = mybg.replace("+", " ")
    #bgic = s.bgicbyq(bg)
    
    return bgic
 return dict(get_bgic = _get_bgic)
    
