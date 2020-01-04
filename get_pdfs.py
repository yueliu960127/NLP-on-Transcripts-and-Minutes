#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:23:55 2019

@author: yueliu
"""
import numpy as np
import urllib
from bs4 import BeautifulSoup
import pandas as pd
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
import pickle
import certifi
'''
Fetch links to all the transcipts from 1982 to 2013
'''
#-------------------------------------------------------------------------------
def get_transcipt_links(years):
    listout = []
    print("----------------------------------------------")
    print("Please Wait, we are fetching the links for you")
    print("----------------------------------------------")
    for year in years:
        print(year)
        home_url = 'https://www.federalreserve.gov/monetarypolicy/fomchistorical'+str(year)+'.htm'
        response = urllib.request.urlopen(home_url, cafile=certifi.where())
        page_source = response.read()
        soup = BeautifulSoup(page_source)
        outdiv = soup.findAll("div", {"class": ["panel panel-default","panel panel-default panel-padded"]})
        for div in outdiv:
            # soupl2 = BeautifulSoup(div)
            headmess = div.findAll("h5")
            if 'Meeting' in str(headmess[0].next) and 'unscheduled' not in str(headmess[0].next):
                temp=div.findAll("a")
                for link in temp:
                    if str(link.contents[0])[0:12]==('Transcript (') :
                        listout.append('https://www.federalreserve.gov'+link.attrs["href"])
    return listout


import requests
requests.packages.urllib3.disable_warnings()

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

#put all links to a list
linklist=get_transcipt_links(np.arange(1982,2014))
# linklist=get_transcipt_links(np.arange(2007,2008))
len(linklist)
#FED error caused a miss in one of the transcripts
linklist[120:129]
#-------------------------------------------------------------------------------
# Check the number of transcripts in each year
# no_of_transcripts_by_year = []
# # for year in np.arange(1982,2014):
# for year in np.arange(1982, 2014):
#     counter = 0
#     for i in linklist:
#         if str(year) in i:
#             counter +=1
#     no_of_transcripts_by_year.append(counter)
#
# df_2 = pd.DataFrame(no_of_transcripts_by_year)
# column_names = ['No_of_pdfs','year']
# df_2['year'] = [year for year in np.arange(1982,2014)]
# df_2.columns = column_names
# df_2


#-------------------------------------------------------------------------------

'''
Download all pdf transcipts to a local folder
'''
#-------------------------------------------------------------------------------
#import wget
#print('Beginning file download with wget module')
#
#for link in linklist:
#    wget.download(link, '/Users/yueliu/Downloads/Fed_links/'+str(link)[56:64]+'.pdf')
#-------------------------------------------------------------------------------


'''
Convert pdf to texts
'''
#-------------------------------------------------------------------------------
import PyPDF2
def get_transcript(transcript_link):
    date = transcript_link.split('/')[5][4:12]
#    response = urllib.request.urlopen(transcript_link)
#    page_source = response.read()
#    soup = BeautifulSoup(page_source, "html5lib")
#    title = soup.find('title').text
#    speech_date = title.split('(', 1)[1].split(')')[0]
#    transcript = soup.find('div', {'id': 'transcript'}).text
#    transcript = transcript.replace('\n', ' ').replace('\r', '').replace('\t', '')
    if 't' in date:
        date = transcript_link.split('/')[5][0:8]
    fname = "/Users/yueliu/Downloads/Fed_links/" + date +'.pdf'
    pdf = open(fname, 'rb')
    transcript = PyPDF2.PdfFileReader(pdf)
    return {
            'date':date,
#            'speaker': speaking,
#            'date': speech_date,
#            'title': title,
            'transcript': transcript
            }
#-------------------------------------------------------------------------------


'''
Make a dictionary for the transcipts
'''
#-------------------------------------------------------------------------------
transcript_dict = {}
for link in enumerate(linklist):
#    if i % 100 == 0:
#        print( 'Scraped ' + str(i) + '/' + str(len(linklist)) + ' of links...')
#    if link.has_attr('href'):
    transcript_data = get_transcript(str(link))
    key = transcript_data['date']
    transcript_dict[key] = transcript_data
#--------------------------------------------------------------------------------------------------------------------------------------------------------------


df = pd.DataFrame.from_dict(transcript_dict, orient='index')

#-------------------------------------------------------------------------------


'''
Lemmatize the 'texts'
'''
#-------------------------------------------------------------------------------
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


#-------------------------------------------------------------------------------
'''
Stop-words
'''
#-------------------------------------------------------------------------------
stop = []
for i in STOPWORDS:
    stop.append(i)

numbers = list(range(0,101))
for i in numbers:
    stop.append(str(i))


stop.extend(['think','federal','meeting','ofthe','january','feburary',
             'march','april','may','june','july','august','september',
             'october','november','december','mr','ms','messrs',
             'actually','have_been','with_other','that_have','will_have',
             'that_there','thank','thank_chairman','yeschairman','that_would',
             'this_year','that_they','next_year','that_will','talk_about',
             'thinking_about','about_percent','this_point','they_have',
             'would_have','talking_about','think_there','would_like','this_meeting',
             'talked_about','there_some','thank_chairman_chairman_bernanke',
             'suggests_that','that_might','some_time','last_year','suggest_that',
             'same_time','board_governor','with_respect','that_were','that_think',
             'sense_that','this_time','think_should','have_some','that_goe','since_last',
             'they_were','little_more','saying_that','over_past','long_term','longer_term',
             'short_term','near_term','suggests_that','fact_that','thinking_about','some_time',
             'believe_that','said_that','something_that','there_been','others_have'])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
texts=[]
texts_multi = []
# for loop starts
for i in range(len(df['transcript'])):
# for i in range(100,101):
#     read_pdf = df['transcript'][i]
    # '''
    # Reads a specific transcipt
    # '''
    # #-------------------------------------------------------------------------------
    # #####################choose how many transcripts to analyze from all the transcripts
    # read_pdf=df['transcript'][289]
    # print(df['date'][289])
    # ###################################################################################
    # #-------------------------------------------------------------------------------

    read_pdf = df['transcript'][i]
    '''
    Transcipt stored in 'texts'
    '''
    #-------------------------------------------------------------------------------
    num_pages = read_pdf.getNumPages()

    ann_text = []
    for page_num in range(num_pages):
        if read_pdf.isEncrypted:
            read_pdf.decrypt("")
            #print(read_pdf.getPage(page_num).extractText())
            page_text = read_pdf.getPage(page_num).extractText().split()
            page_text = simple_preprocess(str(page_text),deacc = True, min_len = 4)
            ann_text.append(page_text)

        else:
            page_text = read_pdf.getPage(page_num).extractText().split()
            page_text = simple_preprocess(str(page_text),deacc = True, min_len = 4)
            ann_text.append(page_text)
    text = ann_text
    #-------------------------------------------------------------------------------

    '''
    Build the bigram and trigram models
    '''
    #-------------------------------------------------------------------------------
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram = gensim.models.Phrases(text, min_count=5, threshold=5) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[text], threshold=5)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)


    def make_bigrams(text):
        return [bigram_mod[doc] for doc in text]

    def make_trigrams(text):
        return [trigram_mod[bigram_mod[doc]] for doc in text]

    data_words_bigrams = make_bigrams(text)
    data_words_trigrams = make_trigrams(text)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # data_lemmatized = lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # text = [[word for word in text if word not in stop] for text in data_lemmatized]
    # text = []
    # for sublist in data_lemmatized:
    #     for word in sublist:
    #         if word not in stop:
    #             text.append(word)
    text_multi = []
    for sublist in data_words_trigrams:
        for word in sublist:
            if word not in stop and '_' in word:
                text_multi.append(word)
    text = []
    for sublist in data_words_trigrams:
        for word in sublist:
            if word not in stop:
                text.append(word)
    texts_multi.append(text_multi)
    texts.append(text)

trans_outfile = 'transcripts_multigrams.pickle'
outfile = open(trans_outfile, "wb" )
pickle.dump(texts_multi, outfile)

trans_outfile = 'transcripts.pickle'
outfile = open(trans_outfile, "wb" )
pickle.dump(texts, outfile)


























