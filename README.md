# Quantext

## Why Quantext?

Quantext quickly extracts insights from student responses to short answer questions or mini-essays to help you assess when and in what ways you could enhance your teaching. 

By using Quantext, material which is routinely uploaded to your institutional Learning Management Systems (LMS) for assessment or plagiarism checks can be systematically analysed to improve teaching and inform learning design. Use Quantext to illuminate student learning and your teaching.

## Who is Quantext for?

Quantext is for educators. 

Quantext has been designed by teachers for teachers working across a wide range of disciplines, class sizes and teaching modes. Quantext can help you to quickly and simply identify and fix ambiguous or confusing questions, address common misconceptions before summative assessments and track how well your students are learning the language of your discipline. Quantext also has potential for analysing student evaluation responses and discussion forum posts.

## Who is using Quantext?

Tertiary educators and academic developers. 

Quantext is in the early stages of development. It is currently being trialled by teachers at several tertiary institutions in NZ and their feedback and input is informing ongoing development. If you would like to contribute to early trials please get in touch - we’d love to hear from you. If you’d just like to see how Quantext works, login to the demo site at <a href="https://quantext.org" target="_blank">https://quantext.org</a> using your Google or Twitter account. You can use the demonstration data set supplied or upload your own data. We have prepared some basic documentation to help get you started. We plan to release Quantext as a cloud-based service in 2018.

## End user documentation
Some basic guides and screencasts to get you started are available at <a href="https://quantext.org/documentation" target="_blank">https://quantext.org/documentation</a>

## Installation & System Requirements
* Quantext has been installed and tested on Mac OSX Sierra 10.12.6 and Ubuntu 16.04
* Quantext source code is released under a GPL 3.0 licence. See<a href="http://www.gnu.org/licenses/" target="_blank">http://www.gnu.org/licenses/</a>.
* The Quantext version on GitHub will not necessarily be the same as the latest production version running on <a href="https://quantext.org" target="_blank">https://quantext.org</a>.
* Please contact us if you would like to get involved in the development effort. <a href="https://quantext.org/contact" target="_blank">https://quantext.org/contact</a>.

### OSX System Requirements

* OSX Sierra 10.12.6+
* Python 3.5.*
* 4Gb RAM
* We recommend the Anaconda python distribution. Anaconda comes with many of the libraries used and is compatible with pip. See <a href="https://www.anaconda.com/download" target="_blank">https://www.anaconda.com/download</a>

#### Installation
* Install Python 3.5+ (Anaconda distribution is easiest)
* Install git - follow instructions <a href="https://gist.github.com/derhuerst/1b15ff4652a867391f03#file-mac-md" target="_blank">here</a> to install Homebrew then use: 
```Shell
brew install git
```
* Install Mongodb as per <a href="https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x" target="_blank">https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x</a>

* Install libraries and Spacy word vector model
```Shell
sudo pip3 install -r requirements.txt
sudo python3 -m spacy download en
```
* Run mongo daemon as per <a href="https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/#run-mongodb" target="_blank">https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/#run-mongodb</a>
* Edit config.py to enter your own authentication details (Google or Twitter)
* Pull Quantext source code from the directory you want to install it into
```Shell
git remote add quantext https://<user>@github.com/quantext/quantext
git pull quantext
cd src
mkdir tmp
cd tmp
touch tft.log
mkdir corpus
mkdir uploads
```
* Launch app - we recommend gunicorn. cd to src directory and run

```Shell
sudo gunicorn --bind 127.0.0.1:8000 tft:app wsgi
```

### Ubuntu System Requirements

* Ubuntu 16.04
* Python 3.5.*
* 4Gb RAM

#### Installation

```Shell
sudo apt-get install python3-pip
sudo apt-get install git
```

Install MongoDB as per <a href="https://docs.mongodb.com/tutorials/install-mongodb-on-ubuntu/" target="_blank">https://docs.mongodb.com/tutorials/install-mongodb-on-ubuntu/</a>

```Shell
sudo pip3 install -r requirements.txt
sudo apt-get install swig
sudo apt-get install libpulse-dev
sudo apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev
sudo pip3 install textract
sudo python3 -m spacy download en

```
* Run mongo daemon as per <a href="https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/#run-mongodb-community-edition" target="_blank">https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/#run-mongodb-community-edition</a>
* Edit config.py to enter your own authentication details (Google or Twitter)
* Pull Quantext source code
```Shell
git remote add quantext https://<user>@github.com/quantext/quantext
git pull quantext
cd src
mkdir tmp
cd tmp
touch tft.log
mkdir corpus
mkdir uploads
```
* Launch app - we recommend gunicorn. cd to src directory and run

```Shell
sudo gunicorn --bind 127.0.0.1:8000 tft:app wsgi


