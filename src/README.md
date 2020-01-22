# Quantext

## Why Quantext?

Quantext quickly extracts insights from student responses to short answer questions or mini-essays to help you assess when and in what ways you could enhance your teaching. 

By using Quantext, material which is routinely uploaded to your institutional Learning Management Systems (LMS) for assessment or plagiarism checks can be systematically analysed to improve teaching and inform learning design. Use Quantext to illuminate student learning and your teaching.

## Who is Quantext for?

Quantext is for educators. 

Quantext has been designed by teachers for teachers working across a wide range of disciplines, class sizes and teaching modes. Quantext can help you to quickly and simply identify and fix ambiguous or confusing questions, address common misconceptions before summative assessments and track how well your students are learning the language of your discipline. Quantext also has potential for analysing student evaluation responses and discussion forum posts.

## Who is using Quantext?

Tertiary educators and academic developers. 

Quantext is in the early stages of development. It is currently being trialled by teachers at several tertiary institutions in NZ and their feedback and input is informing ongoing development. If you would like to contribute to early trials please get in touch - we’d love to hear from you. If you’d just like to see how Quantext works, login using your Google or Twitter account. You can use the demonstration data set supplied or upload your own data. We have prepared some basic documentation to help get you started. We hope to release Quantext as a cloud-based service in 2018.

# System requirements

* Ubuntu 16.04
* Python 3.5.*
* 4Gb RAM

# Installation

```Shell
sudo apt-get install python3-pip
sudo apt-get install git
```

Install MongoDB as per <a href="https://docs.mongodb.com/tutorials/install-mongodb-on-ubuntu/" target="_blank">https://docs.mongodb.com/tutorials/install-mongodb-on-ubuntu/</a>

```Shell
sudo apt-get install python-enchant
sudo pip3 install -r requirements.txt
sudo pip3 install flask_mongoengine
sudo pip3 install oauth2client
sudo pip3 install textacy
sudo apt-get install swig
sudo apt-get install libpulse-dev
sudo apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev
sudo pip3 install textract
sudo python3 -m spacy download en