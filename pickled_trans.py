import pickle
import numpy as np
import pandas as pd
import gensim.corpora as corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import HdpModel
from gensim.models import CoherenceModel
# import pyLDAvis.gensim as gensimvis
# import pyLDAvis
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# import spacy
# nlp = spacy.load('en', disable=['parser', 'ner'])
import os


trans_infile = 'transcripts.pickle'
infile = open(trans_infile, "rb" )
new_texts = pickle.load(infile)

trans_infile = 'transcripts_multigrams.pickle'
infile = open(trans_infile, "rb" )
new_texts = pickle.load(infile)

link_infile = 'link.pickle'
infile = open(link_infile, "rb" )
new_linklist = pickle.load(infile)

# print(new_texts)
# print(type(new_texts))
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
for link in enumerate(new_linklist):
#    if i % 100 == 0:
#        print( 'Scraped ' + str(i) + '/' + str(len(linklist)) + ' of links...')
#    if link.has_attr('href'):
    transcript_data = get_transcript(str(link))
    key = transcript_data['date']
    transcript_dict[key] = transcript_data

df = pd.DataFrame.from_dict(transcript_dict, orient='index')

'''
LDA Model and Coherence Measure
'''
#-------------------------------------------------------------------------------


dictionary = corpora.Dictionary(new_texts)
corpus = [dictionary.doc2bow(text) for text in new_texts]



# transcript_topics = LdaModel(corpus=corpus,
#                              id2word=dictionary,
#                              iterations=500,
#                              num_topics=8,
#                              random_state=100,
#                              update_every=1,
#                              chunksize=100,
#                              passes=10,
#                              alpha='auto',
#                              per_word_topics=True)
#
no_of_topics = 60
#
# transcript_topics = LdaMulticore(corpus=corpus,
#                             num_topics=no_of_topics,
#                             id2word=dictionary,
#                             random_state=100,
#                             chunksize=200,
#                             passes=10,
#                             alpha='symmetric',
#                             decay=0.5,offset=1.0,
#                             eval_every=10,
#                             iterations=50,
#                             gamma_threshold=0.001,
#                             per_word_topics=True)


transcript_topics = HdpModel(corpus,dictionary)
closest_lda = transcript_topics.suggested_lda_model()
# transcript_topics.print_topics(num_topics=20, num_words=100)[1]


# for i, topic in enumerate(transcript_topics.print_topics(no_of_topics)):
#    print('{} --- {}'.format(i, topic))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# extract all document-topic distritbutions to dictionnary

document_key = list(df[transcript_data].index)

document_topic = {}
for doc_id in range(len(corpus)):
    docbok = corpus[doc_id]
    doc_topics = closest_lda.get_document_topics(docbok, 0)
    temp = []
    for topic_id, topic_prob in doc_topics:
        temp.append(topic_prob)
    document_topic[document_key[doc_id]] = temp

# convert dictionnary of document-topic distritbutions to dataframe
df_1 = pd.DataFrame.from_dict(document_topic, orient='index')

# transcripts = [x.split('|')[0] for x in df_1.index]
year_speech = [x for x in df_1.index]

topics_speech = df_1
topic_column_names = ['topic_' + str(i) for i in range(0, 29)]
topics_speech['year'] = pd.Series(year_speech, index=df_1.index)
topic_column_names.append('year')
topics_speech.columns = topic_column_names


columns = ['topic_'+str(i) for i in range(0, 29)] # define columns to process
df_1 = pd.DataFrame(topics_speech.groupby('year')[columns].sum()) # group topics frequency by year
df_1 = 100 * df_1.div(df_1.sum(axis=1), axis=0) # normalize topic frequencies by year
# df_1 = np.round(df_1, 1) # round topic frequencies
df_1.to_csv(r'/Users/yueliu/Desktop/topics_by_year_'+str(29)+'.csv', sep=',')
#-------------------------------------------------------------------------------
fig = plt.figure(figsize=(40,7))
data_perc = df_1/100

# ax1 = fig.add_axes([0.1, 0.3, 0.4, 0.4])
#
# for label in ax1.xaxis.get_ticklabels():
#     # label is a Text instance
#     label.set_color('tab:red')
#     label.set_rotation(45)
#     label.set_fontsize(16)

labels = ['topic_' + str(i) for i in range(26)]
plt.stackplot(year_speech,df_1['topic_1'],df_1['topic_2'],df_1['topic_3'],df_1['topic_4'],df_1['topic_5'],df_1['topic_6'],df_1['topic_7'],df_1['topic_8'],df_1['topic_9'],df_1['topic_10'],df_1['topic_11'],df_1['topic_12'],df_1['topic_13'],df_1['topic_14'],df_1['topic_15'],df_1['topic_16'],df_1['topic_17'],df_1['topic_18'],df_1['topic_19'],df_1['topic_20'],df_1['topic_21'],df_1['topic_22'],df_1['topic_23'],df_1['topic_24'],df_1['topic_25'])
# plt.stackplot(year_speech,df_1['topic_0'],df_1['topic_11'],df_1['topic_15'],df_1['topic_16'],df_1['topic_21'],labels=labels)
plt.legend(loc = 2)
plt.show()






# Coherence Modeling
#-------------------------------------------------------------------------------
coherence_model_lda = CoherenceModel(model=transcript_topics, texts=new_texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#-------------------------------------------------------------------------------


'''
LDA Model visualization
'''
#-------------------------------------------------------------------------------
vis_data = gensimvis.prepare(transcript_topics, corpus, dictionary)

html = pyLDAvis.display(vis_data).data
with open('all_trans.html', 'w') as f:
    f.write(html)
#type(vis_data)
#-------------------------------------------------------------------------------



'''
Optimal number of topics to maximize coherence score
'''
#-------------------------------------------------------------------------------
os.system('afplay /Users/yueliu/Desktop/Notifications/Your_mission.aiff')
os.system('afplay /Users/yueliu/Desktop/Notifications/Mission_Impossible.aiff')
coherence_lda_cv = []
#coherence_lda_umass = []
for i in range(21,25):
    transcript_topics = LdaMulticore(corpus=corpus,
                                     id2word=dictionary,
                                     iterations=50,
                                     num_topics=i,
                                     random_state=100,
                                     chunksize=100,
                                     passes=10,
                                     per_word_topics=True)
    coherence_model_lda_cv = CoherenceModel(model=transcript_topics, corpus = corpus, texts=new_texts, dictionary=dictionary, coherence='c_v')
#    coherence_model_lda_umass = CoherenceModel(model=transcript_topics, corpus = corpus, texts=texts, dictionary=dictionary, coherence='u_mass')
    coherence_lda_cv.append(coherence_model_lda_cv.get_coherence())
#    coherence_lda_umass.append(coherence_model_lda_umass.get_coherence())
os.system('afplay /Users/yueliu/Desktop/Notifications/Mission_accomplished.aiff')

coherence_lda_cv[np.argmax(coherence_lda_cv)]

num_of_topics = np.arange(21,25)
optimal_cv = plt.figure()
optimal_cv = plt.plot(num_of_topics,coherence_lda_cv)
optimal_cv = plt.xlabel('Number of Topics')
optimal_cv = plt.ylabel('Coherence Score')
optimal_cv = plt.title('Coherence Score VS. Number of Topics')
optimal_cv = plt.annotate('maximum coherence score', xy = (num_of_topics[np.argmax(coherence_lda_cv)],coherence_lda_cv[np.argmax(coherence_lda_cv)]))
plt.show()
#-------------------------------------------------------------------------------
