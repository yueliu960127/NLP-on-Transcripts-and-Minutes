import numpy as np
import pandas as pd
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
import os
#-------------------------------------------------------------------------------
all_minutes_files = os.listdir('/Users/yueliu/Desktop/Minutes_texts/')
del(all_minutes_files[39])

def get_minutes(minute):
    date = minute.split("'")[1][0:8]
    print(date)
    minute_file = open('/Users/yueliu/Desktop/Minutes_texts/'+date+'.txt','rb')
    minute_text=minute_file.read()
    minute_file.close()
    return {
            'date':date,
            'minute': minute_text
            }

#-------------------------------------------------------------------------------
'''
Make a dictionary for the Minutes
'''
#-------------------------------------------------------------------------------
minutes_dict = {}
for minute in enumerate(all_minutes_files):
    minutes_data = get_minutes(str(minute))
    key = minutes_data['date']
    minutes_dict[key] = minutes_data

# len(minutes_dict)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

import pickle
df = pd.DataFrame.from_dict(minutes_dict, orient='index')
pickle.dump(df, open( "df_minutes.pickle", "wb" ))

#-------------------------------------------------------------------------------


'''
Lemmatize the 'texts'
'''
#-------------------------------------------------------------------------------
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    doc = nlp(" ".join(texts))
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
             'believe_that','said_that','something_that','there_been','others_have',
             'chairman_bernanke','chairman_greenspan','chairman_volcker'])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
texts=[]
texts_multi = []
# for loop starts
for i in range(len(df['minute'])):
# for i in range(95,101):
    # read_txt = df['minute'][i]
    read_text = df['minute'][i]
    text = simple_preprocess(read_text,deacc = True)
    text = [[word for word in text[100*i:(100*i+100)] if word not in stop] for i in range(int(len(text)/100+1))]
    #-------------------------------------------------------------------------------
    '''
    Build the bigram and trigram models
    '''
    #-------------------------------------------------------------------------------
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram = gensim.models.Phrases(text, min_count=1, threshold=1)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[text], threshold=1)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)


    def make_bigrams(text):
        return [bigram_mod[doc] for doc in text]


    def make_trigrams(text):
        return [trigram_mod[bigram_mod[doc]] for doc in text]


    data_words_bigrams = make_bigrams(text)
    data_words_trigrams = make_trigrams(text)

    # Do lemmatization keeping only noun, adj, vb, adv
    # data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # data_lemmatized = lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # text = [[word for word in text if word not in stop] for text in data_lemmatized]
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



import pickle
minute_outfile = 'minutes_multi.pickle'
outfile = open(minute_outfile, "wb" )
pickle.dump(texts_multi, outfile)

minute_outfile = 'minutes.pickle'
outfile = open(minute_outfile, "wb" )
pickle.dump(texts, outfile)
