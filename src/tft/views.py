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

from flask import render_template, flash, url_for, redirect, request, jsonify, session
from tft import app, db
from werkzeug import secure_filename
from flask_login import login_user, logout_user, current_user
from oauth import OAuthSignIn
from oauth2client.client import OAuth2WebServerFlow
from .models import User, Analysis, RefCorpus, File, Question
import nlp
import numpy

import json
import string
import random
import os
import re
import textract
import textacy

# initialise session vars - not sure this is at all secure!! Need to check :|
def default_settings():
    settings = {}

    settings['nkey'] = app.config['KEYWORDS']
    settings['kblack']= app.config['KEY_BLACKLIST']
    settings['white']= app.config['ONLY_WHITELIST']
    settings['punct']= app.config['FILTER_PUNCT']
    settings['punctuation']=app.config['PUNCTLIST']
    settings['nums']= app.config['FILTER_NUMS']
    settings['measure']= app.config['NG_MEASURE']
    settings['modelanswer'] = app.config['MODEL_ANSWER']
    settings['ncontractions'] = app.config['NORM_CONTRACTIONS']
    settings['trigram'] = app.config['TRIGRAM']
    settings['lcase'] = app.config['LCASE']
    settings['window'] = app.config['WINDOW']
    settings['cgcutoff'] = app.config['CGCUTOFF']
    settings['spell'] = app.config['SPELLCHECK']
    #settings['stem'] = app.config['STEMMING']
    #settings['negreplace'] = app.config['NEGATIVE_REPLACER']
    #settings['synreplace'] = app.config['SYNONYM_REPLACER']
    settings['simalgorithm'] = app.config['SIM_ALGORITHM']

    settings['readability_words'] = app.config['READABILITY_WORDS']
    settings['readability_sents'] = app.config['READABILITY_SENTS']
    settings['readability_ttr'] = app.config['READABILITY_TTR']
    settings['readability_ld'] = app.config['READABILITY_LD']
    settings['readability_smog'] = app.config['READABILITY_SMOG']
    settings['readability_similarity'] = app.config['READABILITY_SIMILARITY']

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

def match_question(file,question_number):
    q = [ques for ques in file.questions if ques.qNum == int(question_number)]
    if q:
        q = q[0]
    return q

@app.route('/')
@app.route('/index')
def index():
    # initialise session vars from config file
    analyses = None
    files = None
    refcorpora = None
    if not current_user.is_anonymous:
        analyses = Analysis.objects(owner=current_user.id).order_by('-created')
        files = File.objects(owner=current_user.id).order_by('-created')
        refcorpora = RefCorpus.objects(owner=current_user.id).order_by('-created')

    return render_template('index.html',
                           title = 'Text Analytics App',
                           analyses = analyses,
                           files = files,
                           refcorpora = refcorpora)

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

def create_demo_files(user):
    s = nlp.SAQData()
    t = nlp.TeachingData()

    filename = "sample.xlsx"
    f = File(filename=filename,owner=user)
    f.save()
    origpath = os.path.join(app.config['CORPUS_FOLDER'],str(f.id))
    if not os.path.exists(origpath):
        os.makedirs(origpath)

    qdata, rdata = s.getandparsedata_sample(filename)

    # save questions
    questions = []
    idx = 1

    for num,text in qdata.items():
        question = Question(qNum=idx,qTitle=num,qText=text)
        qcorpus = s.createqcorpus(rdata[num])
        settings = default_settings()
        question.qSettings = settings
        question.blist = []
        question.wlist = []

        path = os.path.join(origpath,str(idx))
        if not os.path.exists(path):
            os.makedirs(path)

        qcorpus.save(path,name=str(idx),compression="gzip")
        idx = idx + 1
        questions.append(question)

    f.questions = questions
    f.save()

    filenamet = "sample.pdf"
    p = os.path.join(app.config['FILES_FOLDER'],filenamet)
    f = pdf_to_text(p)
    fname = filenamet.rsplit('.', 1)[0]
    filename_new = fname + '.txt'
    fout = open(os.path.join(app.config['FILES_FOLDER'],filename_new),'w')
    fout.write(f)
    fout.close()

    ref = RefCorpus(filename=filenamet,owner=user)
    ref.save()
    t.createtcorpus_sample(filename_new,ref.id)

    return True

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
        create_demo_files(user)

    print("I am saving anew...")
    user.save()
    login_user(user, True)
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    #db.session.rollback()
    return render_template('500.html'), 500

@app.route('/uploadq', methods = ['GET', 'POST'])
def upload_file_qs():
    s = nlp.SAQData()
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
                f = File(filename=filename,owner=user)
                f.save()

                origpath = os.path.join(app.config['CORPUS_FOLDER'],str(f.id))
                if not os.path.exists(origpath):
                    os.makedirs(origpath)

                # do the processing here on upload
                qdata, rdata = s.getandparsedata(filename)

                # save questions
                questions = []
                idx = 1

                for num,text in qdata.items():
                    question = Question(qNum=idx,qTitle=num,qText=text)
                    qcorpus = s.createqcorpus(rdata[num])
                    settings = default_settings()
                    question.qSettings = settings
                    question.blist = []
                    question.wlist = []

                    path = os.path.join(origpath,str(idx))
                    if not os.path.exists(path):
                        os.makedirs(path)

                    qcorpus.save(path,name=str(idx),compression="gzip")
                    idx = idx + 1
                    questions.append(question)

                f.questions = questions
                f.save()

                return redirect(url_for('index'))
            except Exception as e:
                flash('There is a problem uploading your file: ' + str(e))
            return redirect(url_for('index'))

        else:
            flash('Your file must have a .xls or .xlsx extension')

    return redirect(url_for('index'))

@app.route('/uploadt', methods = ['GET', 'POST'])
def upload_file_teach():
#This is bare minimum function to upload teaching materials
    t = nlp.TeachingData()
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_filet(f.filename):
            try:
                filenamet = fname_generator(secure_filename(f.filename))
                f.save(os.path.join(app.config['UPLOAD_FOLDER'],filenamet))
                user = User.objects.get(id=current_user.id)

                if filenamet.rsplit('.', 1)[1] == 'pdf':     #txt file is just saved. Saved pdf file needs converted to txt
                    p = os.path.join(app.config['UPLOAD_FOLDER'],filenamet)
                    f = pdf_to_text(p)
                    fname = filenamet.rsplit('.', 1)[0]
                    filename_new = fname + '.txt'
                    fout = open(os.path.join(app.config['UPLOAD_FOLDER'],filename_new),'w')
                    fout.write(f)
                    fout.close()

                    ref = RefCorpus(filename=filenamet,owner=user)
                    ref.save()

                    t.createtcorpus(filename_new,ref.id)

                return redirect(url_for('index'))
            except Exception as e:
                flash('There is a problem uploading your file: ' + str(e))
            return redirect(url_for('index'))
        else:
            flash('Your file must have a .pdf or .txt extension')

    return redirect(url_for('index'))

@app.route('/newAnalysis', methods = ['POST'])
def new_analysis():
    files = request.form.get('files')
    name = request.form.get('name')
    ref = request.form.get('refcorpus')
    user = User.objects.get(id=current_user.id)
    analysis = Analysis(owner=user)
    analysis.name = name

    if files:
        file_list = files.split(",")
        student_files = File.objects(id__in=file_list)
        analysis.files = student_files

    if ref:
        ref_list = ref.split(",")
        ref_corpus = RefCorpus.objects(id__in=ref_list)
        analysis.refcorpus = ref_corpus

    analysis.save()

    return str(analysis.id)

@app.route('/deleteAnalysis', methods = ['POST'])
def delete_analysis():
    analysis_id = request.form.get('id')
    an = Analysis.objects.get(id=analysis_id)
    an.delete()
    return str(an.id)

def build_student_file(analysis_id,file,question,left_or_right):
    analysis = Analysis.objects.get(id=analysis_id)
    qcorpus = load_corpus(file,question)
    anaq = ana_q(qcorpus,question)
    sumq = sum_q(qcorpus,question['qSettings']['modelanswer'])

    student_file = {}
    student_file["file"] = file
    student_file["q"] = question
    student_file["corpus"] = qcorpus
    student_file["anaq"] = anaq
    student_file["sumq"] = sumq
    student_file["left_or_right"] = left_or_right
    student_file["total"] = len(analysis.files)
    student_file["index"] = analysis.files.index(file)
    student_file["an"] = analysis

    fullcorpus = None
    t = nlp.TeachingData()

    for q in file.questions:
        qcorpus = load_corpus(file,q)
        if fullcorpus is None:
            fullcorpus = qcorpus
        else:
            for doc in qcorpus.docs:
                fullcorpus.add_text(doc.text, doc.metadata)

    chars,totalwords,unique_words,totalsents,lexicaldiversity,lexicaldensity,smog,gunning,flesch_ease,fk = t.summarisetcorpus(fullcorpus)
    student_file["full_corpus"] = {"Characters":chars,"Words":totalwords,"Unique":unique_words,"Sentences":totalsents,"TTR":lexicaldiversity,"LD":lexicaldensity,"SMOG":smog,"Gunning":gunning,"Flesch":flesch_ease,"FK":fk}
    return student_file

@app.route('/loadStudentFile', methods = ['POST'])
def load_student_file():
    analysis_id = request.form.get('analysis_id')
    file_id = request.form.get('file_id')
    left_or_right = request.form.get('left_or_right')

    file = File.objects.get(id=file_id)
    question = match_question(file,1)
    student_file = build_student_file(analysis_id,file,question,left_or_right)

    return render_template('studentFile.html',
                           student_file=student_file)

@app.route('/get_chart_data', methods = ['GET'])
def get_chart_data():

    analysis_id = request.args.get('analysis_id')
    f1 = request.args.get('f1')
    f2 = request.args.get('f2')
    q1 = request.args.get('q1')
    q2 = request.args.get('q2')

    an = Analysis.objects.get(id=analysis_id)
    file_array = [f1,f2]
    question_array = [q1,q2]
    chartdata = []

    for n in range(0,2):
        file = File.objects.get(id=file_array[n])
        question = match_question(file,question_array[n])
        qcorpus = load_corpus(file,question)
        sumq = sum_q(qcorpus,question['qSettings']['modelanswer'])
        sumq["index"] = an.files.index(file)+1
        chartdata.append(sumq)

    chartdata = json.dumps(chartdata)
    return chartdata

@app.route('/changeQuestion', methods = ['POST'])
def change_question():
    student_file = reprocessStudentFile(request.form)
    return render_template('studentFile.html',
                           student_file=student_file)

@app.route("/save_theme", methods = ['POST'])
def save_theme():
    theme = request.form.get("theme")
    if not current_user.is_anonymous:
        current_user.theme = theme
        current_user.save()
    return "Success"

@app.route('/analyse/', methods = ['GET'])
@app.route('/analyse/<analysis_id>', methods = ['GET'])
def analyse(analysis_id=None):
    if not current_user.is_anonymous:
        an = None
        student_files = []
        if not analysis_id is None:
            an = Analysis.objects.get(id=analysis_id)

            for idx in range(0,2):
                if len(an.files) > idx:
                    left_or_right = "sLeft" if idx == 0 else "sRight"
                    file_id = an.files[idx]["id"]
                    file = File.objects.get(id=file_id)
                    question = match_question(file,1)
                    student_file = build_student_file(analysis_id,file,question,left_or_right)
                    student_files.append(student_file)

            if an.refcorpus:
                for r in an.refcorpus:
                    tcorpus = load_tcorpus(r)
                    t_kwords,t_bgs,warning,means = ana_t(tcorpus)
                    r.t_kwords = t_kwords
                    r.t_bgs = t_bgs
                    r.means = means

        return render_template('analyse.html',
                               user=current_user,
                               an=an,
                               analysis_id=analysis_id,
                               student_files=student_files
                               )
    else:
        return redirect(url_for('index'))

#@app.route('/visualise')
#@app.route('/visualise/<question>')
#@app.route('/visualise/<question>/<ngic>')
#def visualise(question=None, ngic=None):
#     return render_template('visualise.html', question=question, ngic=ngic)

@app.route('/charts/<chartdata>/<charttype>')
def charts(chartdata=None, charttype=None):
    return render_template('charts.html', chartdata=chartdata, charttype=charttype)

@app.route('/development')
def development():
    return render_template('development.html', title = 'Development')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html', title = 'Documentation')

@app.route('/contact')
def contact():
    return render_template('contact.html', title = 'Contact Quantext')

def reprocessStudentFile(form):
    file_id = form.get('id')
    question_number = form.get('question')
    analysis_id = form.get('analysis_id')
    left_or_right = form.get('left_or_right')

    file = File.objects.get(id=file_id)
    question = match_question(file,question_number)
    key = str(request.form.get("key"))
    val = str(request.form.get("val"))

    if key is not None and val is not None:
        if type(val) == str:
            if val == "True":
                val = True
            elif val == "False":
                val = False
            elif re.match(r'\d+',val):
                val = int(val)
        question['qSettings'][key] = val

    question.save()

    student_file = build_student_file(analysis_id,file,question,left_or_right)
    return student_file

@app.route('/settings', methods = ['GET', 'POST'])
def settings():
    student_file = reprocessStudentFile(request.form)
    return render_template('keywords.html',
                           student_file=student_file)

@app.route('/get_kwic', methods = ['POST'])
def get_kwic():
    analysis_id = request.form.get("analysis_id")
    myword = request.form.get("myword")
    an = Analysis.objects.get(id=analysis_id)

    if an.refcorpus:
        for r in an.refcorpus:
            tcorpus = load_tcorpus(r)
            t_kwords,t_bgs,warning,means = ana_t(tcorpus)
            r.means = means
            r.kwic = return_kwic(tcorpus,myword)

    return render_template('kwic.html', an = an)

@app.route('/get_wordtree', methods = ['POST'])
def get_wordtree():
    file_id = request.form.get("file_id")
    question_number = request.form.get("question_number")
    file = File.objects.get(id=file_id)
    question = match_question(file,question_number)
    qcorpus = load_corpus(file,question)
    myword = request.form.get("myword")
    kwic = return_kwic(qcorpus,myword)
    return json.dumps(kwic)

def add_new_word(request_form,key):
    file_id = request_form.get('id')
    question_number = request_form.get('question')
    word = request.form.get('word')

    file = File.objects.get(id=file_id)
    question = match_question(file,question_number)

    if question[key]:
        question[key].append(word)
    else:
        question[key] = [word]

    question.save()
    return file, question, word

@app.route('/add_category', methods = ['POST'])
def add_category():
    add_new_word(request.form,"categories")
    return "Success"

@app.route('/add_blist', methods=['POST'])
def add_blist():
    file, question, word = add_new_word(request.form,"blist")
    analysis_id = request.form.get('analysis_id')
    left_or_right = request.form.get('left_or_right')
    student_file = build_student_file(analysis_id,file,question,left_or_right)
    return render_template('blist.html',
                           student_file=student_file,
                           item=word
                           )

@app.route('/save_notes', methods = ['GET', 'POST'])
def save_notes():
    file_id = request.form.get("id")
    question_number = request.form.get("question")
    notes = request.form.get("notes")
    response = request.form.get("response")
    file = File.objects.get(id=file_id)
    question = match_question(file, question_number)
    qcorpus = load_corpus(file,question)

    match_func = lambda doc: doc.metadata["ID"] == response
    for doc in qcorpus.get(match_func):
        doc.metadata["notes"] = notes

    origpath = os.path.join(app.config['CORPUS_FOLDER'],str(file.id))
    path = os.path.join(origpath,str(question_number))
    qcorpus.save(path,name=str(question_number),compression="gzip")
    return "Success"

@app.route('/save_category', methods = ['GET', 'POST'])
def save_category():
    file_id = request.form.get("id")
    question_number = request.form.get("question")
    category = request.form.get("category")
    response = json.loads(request.form.get("response"))
    file = File.objects.get(id=file_id)
    question = match_question(file, question_number)
    qcorpus = load_corpus(file,question)

    for r in response:
        match_func = lambda doc: doc.metadata["ID"] == r
        for doc in qcorpus.get(match_func):
            if category not in doc.metadata["categories"]:
                doc.metadata["categories"].append(category)

    origpath = os.path.join(app.config['CORPUS_FOLDER'],str(file.id))
    path = os.path.join(origpath,str(question_number))
    qcorpus.save(path,name=str(question_number),compression="gzip")
    return "Success"

@app.route('/remove_category', methods = ['GET', 'POST'])
def remove_category():
    file_id = request.form.get("id")
    question_number = request.form.get("question")
    category = request.form.get("category")
    response = json.loads(request.form.get("response"))
    file = File.objects.get(id=file_id)
    question = match_question(file, question_number)
    qcorpus = load_corpus(file,question)

    for r in response:
        match_func = lambda doc: doc.metadata["ID"] == r
        for doc in qcorpus.get(match_func):
            if category in doc.metadata["categories"]:
                doc.metadata["categories"].remove(category)

    origpath = os.path.join(app.config['CORPUS_FOLDER'],str(file.id))
    path = os.path.join(origpath,str(question_number))
    qcorpus.save(path,name=str(question_number),compression="gzip")
    return "Success"

@app.route('/delete_category', methods=['POST'])
def delete_category():
    file_id = request.form.get('id')
    question_number = request.form.get('question')
    word = request.form.get('word')

    file = File.objects.get(id=file_id)
    question = match_question(file,question_number)

    question['categories'].remove(word)
    question.save()
    return "Success"

@app.route('/delete_blist', methods=['POST'])
def delete_blist():
    file_id = request.form.get('id')
    question_number = request.form.get('question')
    word = request.form.get('word')

    file = File.objects.get(id=file_id)
    question = match_question(file,question_number)

    question['blist'].remove(word)
    question.save()
    return "Success"

def load_corpus(file,question):
    s = nlp.SAQData()
    path = os.path.join(app.config['CORPUS_FOLDER'],str(file.id),str(question.qNum))
    qcorpus = s.loadcorpus(path,question)
    return qcorpus

def load_tcorpus(ref):
    t = nlp.TeachingData()
    filename = ref.filename.rsplit('.', 1)[0]
    path = os.path.join(app.config['CORPUS_FOLDER'],str(ref.id),filename)
    tcorpus = t.loadcorpus(path,filename)
    return tcorpus

def sum_q(qcorpus,modelanswer):
    #summarise responses by length and extract key summary stats
    s = nlp.SAQData()
    ws = s.get_worksheet(qcorpus.docs,modelanswer)
    means = s.get_means(ws)
    qcount = len(ws)
    chartdata = get_chartdata(ws)
    hist, values = numpy.histogram(ws["Sentences"].values)
    return dict(qcount=qcount,means=means,chartdata=chartdata,sents=hist.tolist())

def create_chart(wsdata, chartdata, ws_key):
    obj = {}
    maxval = max(wsdata[ws_key].values)
    minval = min(wsdata[ws_key].values)

    if isinstance(maxval,(numpy.int64, numpy.integer)):
        if abs(maxval - minval) < 20:
            bins = range(minval, maxval + 2, 1)
        else:
            binwidth = int((maxval - minval)/10)
            bins = range(minval, maxval + binwidth, binwidth)
            if not binwidth == 0:
                hist, values = numpy.histogram(wsdata[ws_key].values, bins=bins)

                for index,value in numpy.ndenumerate(hist):
                    obj[str(values[index])] = int(value)

    elif isinstance(maxval,(numpy.float64, numpy.float)):
        binwidth = (maxval-minval)/10
        bins=numpy.arange(minval, maxval + binwidth, binwidth)
        if not binwidth == 0:
            hist, values = numpy.histogram(wsdata[ws_key].values, bins=bins)

            for index,value in numpy.ndenumerate(hist):
                obj[str(values[index])] = int(value)

    chartdata[ws_key] = obj
    return chartdata

def get_chartdata(wsdata):
    chartdata = create_chart(wsdata,{},"Words")
    chartdata = create_chart(wsdata,chartdata,"Sentences")
    chartdata = create_chart(wsdata,chartdata,"TTR")
    chartdata = create_chart(wsdata,chartdata,"LD")
    chartdata = create_chart(wsdata,chartdata,"SMOG")
    chartdata = create_chart(wsdata,chartdata,"Similarity")
    return chartdata

def ana_q(qcorpus,q):
    s = nlp.SAQData()
    top_kwords, top_bgs, warning = s.analyseqcorpus(qcorpus,q)
    return {"top_kwords":top_kwords, "top_bgs":top_bgs, "warning":warning}

def ana_t(tcorpus):
    t = nlp.TeachingData()
    chars,words,unique_words,sentences,ttr,ld,smog,gunning,flesch_ease,fk = t.summarisetcorpus(tcorpus)
    means = {"Characters":chars,"Words":words,"Unique":unique_words,"Sentences":sentences,"TTR":ttr,"LD":ld,"SMOG":smog,"Gunning":gunning,"Flesch":flesch_ease,"FK":fk}
    top_kwords,top_bgs,warning = t.analysetcorpus(tcorpus,default_settings())
    return top_kwords,top_bgs,warning,means

def return_kwic(corpus, myword):
    s = nlp.SAQData()
    return s.kwicbyq(corpus,myword)

#API methods
@app.route('/datatables/<file_id>/<question_number>', methods=['GET'])
@app.route('/datatables/<file_id>/<question_number>/<search>', methods=['GET'])
def get_datatables(file_id, question_number, search=None):
    s = nlp.SAQData()
    file = File.objects.get(id=file_id)
    question = match_question(file,question_number)
    qcorpus = load_corpus(file,question)
    if search:
        qcorpus = qcorpus.get(lambda x: bool(re.search(r'\b%s\b' % search,x.text)))
    else:
        qcorpus = qcorpus.docs

    ws = s.get_worksheet(qcorpus,question['qSettings']['modelanswer'],search)
    return ws.to_json(orient='table')

@app.route('/similarity/<file_id>/<question_number>/', methods=['POST'])
def similarity(file_id, question_number):
    reference_answer = request.form.get('reference_answer')
    s = nlp.SAQData()
    file = File.objects.get(id=file_id)
    question = match_question(file,question_number)
    question["qSettings"]["modelanswer"] = reference_answer
    question.save()
    qcorpus = load_corpus(file,question)
    qcorpus = qcorpus.docs
    ws = s.get_worksheet(qcorpus,reference_answer,None)
    chartdata = get_chartdata(ws)
    sumq = { "chartdata":chartdata, "ws":ws.to_json(orient='table') }
    return json.dumps(sumq)

@app.route('/reference_answer/<file_id>/<question_number>', methods=['POST'])
def reference_answer(file_id, question_number):
    reference_answer = request.form.get('reference_answer')
    file = File.objects.get(id=file_id)
    question = match_question(file,question_number)
    if reference_answer:
        if question.referenceAnswers:
            question.referenceAnswers.append(reference_answer)
        else:
            question.referenceAnswers = [reference_answer]
        question.save()
    return "Success"