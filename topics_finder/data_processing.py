from abc import ABC, abstractmethod
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from .utils import open_data


STEMMER = PorterStemmer()


class BasePreparationClass(ABC):

    @abstractmethod
    def process_file(self):
        """
        All the models expect Preparation Classes to return list of lists of words.
        Example: [['fire', 'wit', 'must', 'awar', 'defam'], ['call', 'infrastructur', 'protect', 'summit']]

        :return: List of lists of words
        """
        pass


class DataPreparation(BasePreparationClass):
    """
    Default DataPreparation class that works with the tgz file in data/
    """
    def __init__(self, stop_words=None):
        self.file = list(open_data())
        if not stop_words:
            self.stop_words = set(stopwords.words('english'))

    def process_file(self):
        # drop stop words, lemmatize and stemming
        processed_object = list()
        for txt_file in self.file:
            temp_processed_file = self._preprocessing(txt_file)
            processed_object.append(temp_processed_file)
        return processed_object

    def _preprocessing(self, txt_file):
        """
        Do the following preprocessing:
        - Drop stop_words
        - Drop small words (lower than 3)
        - Run lemmatize and stemming

        :param txt_file: a txt file
        :return: list of processed words
        """
        result = []
        for token in word_tokenize(txt_file):
            if token not in self.stop_words and len(token) > 3:
                result.append(self._lemmatize_stemming(token))
        return result

    @staticmethod
    def _lemmatize_stemming(text):
        return STEMMER.stem(WordNetLemmatizer().lemmatize(text, pos='v'))




