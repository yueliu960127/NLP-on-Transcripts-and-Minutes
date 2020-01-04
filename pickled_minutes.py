import pickle
import numpy as np
import pandas as pd
import gensim.corpora as corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from gensim.models import HdpModel
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import hlda
from hlda.sampler import HierarchicalLDA

minute_infile = 'minutes.pickle'
infile = open(minute_infile, "rb" )
new_texts = pickle.load(infile)

minute_infile = 'minutes_multi.pickle'
infile = open(minute_infile, "rb" )
new_texts = pickle.load(infile)


df = pickle.load( open( "df_minutes.pickle", "rb" ))

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

minute_topics = HdpModel(corpus,dictionary)

closest_lda = minute_topics.suggested_lda_model()

# minute_topics = LdaMulticore(corpus=corpus,
#                              id2word=dictionary,
#                              iterations=500,
#                              num_topics=no_of_topics,
#                              random_state=100,
#                              chunksize=100,
#                              passes=10,
#                              per_word_topics=True)
#
# minute_topics.print_topics(num_topics=59, num_words=100)[1]
#
# for i, topic in enumerate(minute_topics.print_topics(20)):
#    print('{} --- {}'.format(i, topic))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# extract all document-topic distritbutions to dictionnary
document_key = list(df['minute'].index)

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
topic_column_names = ['topic_' + str(i) for i in range(0, 83)]
topics_speech['year'] = pd.Series(year_speech, index=df_1.index)
topic_column_names.append('year')
topics_speech.columns = topic_column_names


columns = ['topic_'+str(i) for i in range(0, 83)] # define columns to process
df_1 = pd.DataFrame(topics_speech.groupby('year')[columns].sum()) # group topics frequency by year
df_1 = 100 * df_1.div(df_1.sum(axis=1), axis=0) # normalize topic frequencies by year
# df_1 = np.round(df_1, 1) # round topic frequencies
df_1.to_csv(r'/Users/yueliu/Desktop/topics_by_year_minutes_83.csv', sep=',')

#-------------------------------------------------------------------------------
#
#
# Coherence Modeling
#-------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------
#
#
# # '''
# # LDA Model visualization
# # '''
# # #-------------------------------------------------------------------------------
# # vis_data = gensimvis.prepare(minute_topics, corpus, dictionary)
# #
# # html = pyLDAvis.display(vis_data).data
# # with open('20060629.html', 'w') as f:
# #     f.write(html)
# # #type(vis_data)
# # #-------------------------------------------------------------------------------
#
#
#
'''
Optimal number of topics to maximize coherence score
'''
#-------------------------------------------------------------------------------

coherence_lda_cv = []
#coherence_lda_umass = []
for i in range(16,30):
    transcript_topics = LdaMulticore(corpus=corpus,
                                     id2word=dictionary,
                                     iterations=500,
                                     num_topics=i,
                                     random_state=100,
                                     chunksize=100,
                                     passes=10,
                                     per_word_topics=True)
    coherence_model_lda_cv = CoherenceModel(model=transcript_topics, corpus = corpus, texts=new_texts, dictionary=dictionary, coherence='c_v')
#    coherence_model_lda_umass = CoherenceModel(model=transcript_topics, corpus = corpus, texts=texts, dictionary=dictionary, coherence='u_mass')
    coherence_lda_cv.append(coherence_model_lda_cv.get_coherence())
#    coherence_lda_umass.append(coherence_model_lda_umass.get_coherence())

coherence_lda_cv[np.argmax(coherence_lda_cv)]

num_of_topics = np.arange(16,30)
optimal_cv = plt.figure()
optimal_cv = plt.plot(num_of_topics,coherence_lda_cv)
optimal_cv = plt.xlabel('Number of Topics')
optimal_cv = plt.ylabel('Coherence Score')
optimal_cv = plt.title('Coherence Score VS. Number of Topics')
optimal_cv = plt.annotate('maximum coherence score', xy = (num_of_topics[np.argmax(coherence_lda_cv)],coherence_lda_cv[np.argmax(coherence_lda_cv)]))
plt.show()
#-------------------------------------------------------------------------------
