from flask import render_template, flash, url_for, redirect, request, jsonify
from tft import app, db
from werkzeug import secure_filename
#from .forms import LoginForm
from flask_login import login_user, logout_user, current_user
from oauth import OAuthSignIn
from models import User


import string
import random
import os
import re
import nlp
import pandas as pd

#initialise question data structure variable - this is set when the question file successfully uploaded 
SAQDATA = None
MODELANSWER = "Enter Model Answer"

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


@app.route('/')
@app.route('/index')
@app.route('/index/<filename>')
@app.route('/index/<filename>/<filenamet>')
def index(filename=None, filenamet=None):
    
    if filename and not SAQDATA:
        
        get_saq_data(filename)
   
    return render_template('index.html',
                           title = 'Text Analytics App',
                           filename = filename,
                           filenamet = filenamet)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/authorize/<provider>')
def oauth_authorize(provider):
    if not current_user.is_anonymous:
        return redirect(url_for('index'))
    oauth = OAuthSignIn.get_provider(provider)
    return oauth.authorize()

@app.route('/callback/<provider>')
def oauth_callback(provider):
    if not current_user.is_anonymous:
        return redirect(url_for('index'))
    oauth = OAuthSignIn.get_provider(provider)
    social_id, username, email = oauth.callback()
    if social_id is None:
        flash('Authentication failed.')
        return redirect(url_for('index'))
    user = User.query.filter_by(social_id=social_id).first()
    if not user:
        user = User(social_id=social_id, nickname=username, email=email)
        db.session.add(user)
        db.session.commit()
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
                   return redirect(url_for('index', filename=filename))
               except Exception as e:
                   flash('There is a problem uploading your file: ' + str(e))
               return redirect(url_for('index'))

         else:

               flash('Your file must have a .xls or .xlsx extension')

        



   return redirect(url_for('index'))
   
@app.route('/uploadt/<filename>', methods = ['GET', 'POST'])
def upload_file_teach(filename=None):
#This is bare minimum function to upload teaching materials

   if request.method == 'POST':
             
         f = request.files['file']

         if f and allowed_filet(f.filename):
               try:
                   filenamet = fname_generator(secure_filename(f.filename))
                   f.save(os.path.join(app.config['UPLOAD_FOLDER'],filenamet))
                   return redirect(url_for('index', filename=filename, filenamet=filenamet))
               except Exception as e:
                   flash('There is a problem uploading your file: ' + str(e))
               return redirect(url_for('index', filename=filename))

         else:

               flash('Your file must have a .pdf or .txt extension')

        



   return redirect(url_for('index', filename=filename))

 
@app.route('/analyse/<filename>/<window>', methods = ['GET', 'POST'])
@app.route('/analyse/<filename>/<question>/<window>', methods = ['GET', 'POST'])
@app.route('/analyse/<filename>/<question>/<ngic>/<window>', methods = ['GET', 'POST'])
@app.route('/analyse/<filename>/<filenamet>/<window>', methods = ['GET', 'POST'])
@app.route('/analyse/<filename>/<filenamet>/<question>/<window>', methods = ['GET', 'POST'])
@app.route('/analyse/<filename>/<filenamet>/<question>/<ngic>/<window>', methods = ['GET', 'POST'])
def analyse(filename=None, filenamet=None, question=None, ngic=None, window=None, ):
    if not question:
        question = request.form.get('selectq') 
        
    if not filenamet:
        filenamet = 'no-teaching-context'
    
    window = int(window)
    if request.form.get('window'):
         
        window = int(request.form.get('window'))


    return render_template('analyse.html', filename=filename, filenamet=filenamet, question=question, ngic=ngic,  window=window)   
     
     
@app.route('/visualise')   
@app.route('/visualise/<filename>/<question>')
@app.route('/visualise/<filename>/<question>/<ngic>')
def visualise(filename=None, question=None, ngic=None):


     return render_template('visualise.html', filename=filename, question=question, ngic=ngic)   
     

@app.route('/about')
def about():
   
    return render_template('about.html',
                           title = 'About Text Analytics App')
    
@app.route('/contact')
def contact():
   
    return render_template('contact.html',
                           title = 'Contact InClass Team')
    
@app.route('/settings')
def settings():
   
    return render_template('settings.html',
                           title = 'Text Analytics Settings')
    

@app.route('/_process_model_answer')
def process_model_answer():
    global MODELANSWER
    try:
        answer = request.args.get('modelanswer')            
        MODELANSWER = str(answer) #set global MODELANSWER
        return jsonify(result=MODELANSWER)
    except Exception as e:
        return str(e)
  
@app.route('/_key_display')
def key_display():
	try:
         numKeywords = request.args.get('numKeywords')
         app.config['KEYWORDS'] = numKeywords
         return jsonify(result=str(numKeywords))
	except Exception as e:
		return str(e)
    
    
def get_saq_data(filename):
    "Given a filename, parse file and set global SAQDATA "
    global SAQDATA
    SAQDATA = nlp.SAQData(filename)
    return
 
@app.template_filter('getqlist')
def get_qs(filename):
    saqdata = SAQDATA
    qlist = sort_ids(saqdata.pdata.keys())    
    
    return qlist

    
@app.context_processor
def get_model_answer():

    return dict(get_model_answer = MODELANSWER)
    
@app.context_processor
def sum_q():
 def _sum_q(filename, question):
    
    saqdata = SAQDATA
    #extract question text
    qtext = saqdata.pdata[question][0]
    
    #summarise responses by length and extract key summary stats
    txt, stats = saqdata.summarisedatabyq(saqdata.pdata, question)
    txt = pd.DataFrame(txt)

    txt = zip(txt['Response'],txt['Length'])
    stats = pd.DataFrame(stats)
    
    kstats = {}
    kstats['count'] = int(stats.ix['count', 'Length'])
    kstats['mean'] = int(stats.ix['mean', 'Length'])
    kstats['median'] = int(stats.ix['50%', 'Length'])
    kstats['min'] = int(stats.ix['min', 'Length'])
    kstats['max'] = int(stats.ix['max', 'Length'])

    return qtext, txt, kstats
 return dict(sum_q = _sum_q)


@app.context_processor
def ana_q():
 def _ana_q(filename, question, window=2):
    
    saqdata = SAQDATA
    qcorpus = saqdata.normalisedata(saqdata.pdata, question)
    top_kwords, top_bgs = saqdata.analysedatabyq(qcorpus, window)
    return top_kwords, top_bgs, qcorpus 
 return dict(ana_q = _ana_q)
 
 
@app.context_processor
def get_kwic():
 def _get_kwic(filename, qcorpus, myword):
    
    saqdata = SAQDATA
    
    kwic = saqdata.kwicbyq(qcorpus, myword)
    kwic_complete = [[line[1] + ' ' + line[2] + ' ' + line[3]] for line in kwic] #required format for Wordtree visualisation
    
    return kwic, kwic_complete
 return dict(get_kwic = _get_kwic)

@app.context_processor
def get_bgic():
 def _get_bgic(filename, qcorpus, mybg, window):
    
    saqdata = SAQDATA
    bg = tuple(mybg.split('+'))
    bgic = saqdata.bgicbyq(qcorpus, bg, window)
    
    return bgic
 return dict(get_bgic = _get_bgic)
    