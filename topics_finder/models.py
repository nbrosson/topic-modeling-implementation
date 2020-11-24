import numpy as np

from abc import ABC, abstractmethod
from .data_processing import DataPreparation
from gensim.corpora import Dictionary
from gensim import models
from scipy import linalg
from sklearn.feature_extraction.text import CountVectorizer


class BaseModel(ABC):
    def __init__(self, nb_topics, data_cleanser):
        self.nb_topics = nb_topics
        self.cleansed_data = data_cleanser.process_file()

    @abstractmethod
    def train_predict(self):
        pass


class LsaModel(BaseModel):
    def __init__(self, nb_topics=10, data_cleanser=DataPreparation()):
        BaseModel.__init__(self, nb_topics, data_cleanser)

    def train_predict(self):
        bow_corpus, dictionary = self._feature_preparations()
        lda_model = models.LdaMulticore(bow_corpus, num_topics=self.nb_topics, id2word=dictionary, passes=2, workers=2)
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))

    def _feature_preparations(self):
        dictionary = Dictionary(self.cleansed_data)

        # keep tokens that appear in more than 5 documents; just keep the first 90000 most frequent tokens
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=90000)

        # For each document we create a dictionary reporting how many words and how many times those words appear.
        bow_corpus = [dictionary.doc2bow(doc) for doc in self.cleansed_data]
        return bow_corpus, dictionary


class LsaModelWithTfIdf(LsaModel):

    def __init__(self, nb_topics=10):
        LsaModel.__init__(self, nb_topics)

    def train_predict(self):
        bow_corpus, dictionary = self._feature_preparations()
        corpus_tfidf = models.TfidfModel(bow_corpus)[bow_corpus]
        lda_model_tfidf = \
            models.LdaMulticore(corpus_tfidf, num_topics=self.nb_topics, id2word=dictionary, passes=2, workers=2)
        for idx, topic in lda_model_tfidf.print_topics(-1):
            print('Topic: {} Word: {}'.format(idx, topic))


class SvdModel(BaseModel):
    def __init__(self, nb_topics=10, data_cleanser=DataPreparation()):
        BaseModel.__init__(self, nb_topics, data_cleanser)

    def train_predict(self):
        Vh, vectorizer = self._feature_preparations()
        vocab = np.array(list(vectorizer.vocabulary_.keys()))
        print(self._show_topics(Vh[:self.nb_topics], vocab))

    def _feature_preparations(self):
        # we need a list of strings instead of a list of lists of words
        words_array = [" ".join(list_of_words) for list_of_words in self.cleansed_data]
        vectorizer = CountVectorizer()
        features_object = vectorizer.fit_transform(words_array).todense()
        U, s, Vh = linalg.svd(features_object, full_matrices=False)
        return Vh, vectorizer

    @staticmethod
    def _show_topics(a, vocab):
        """
        a: Matrix with Topics as rows, and words as columns. each value is a coef, representing
        the weight of the word for the topic.

        -> Returns the top num_top_words words for each topic.
        """
        num_top_words = 8

        # t is sorted by the columns highest values. we select the NUM TOP WORDS first values
        top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words - 1: -1]]
        topic_words = ([top_words(t) for t in a])  # t: topic; for each row of a, run top_words
        return [' '.join(t) for t in topic_words]