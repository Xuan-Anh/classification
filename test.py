class Filter_Sport():
    from xgboost.sklearn import XGBClassifier
    from gensim.models import KeyedVectors
    import numpy as np
    from pyvi import ViTokenizer
    import re

    def init(self, path_vector_transformer: str, path_vocabulary: str, path_model: str):
        self._model = self.XGBClassifier()
        self._model.load_model(path_model)
        with open(path_vocabulary) as f:
            self._vocab = f.read().splitlines()
        self._w2v = self.KeyedVectors.load(path_vector_transformer)

    def predict(self, text):
        temp = self.re.sub(r"http\S+", "", text)
        temp = self.re.sub("xa0", " ", temp)
        temp = "".join([cha.lower() if cha.isalpha() or cha == " " else "" for cha in temp])
        temp = self.ViTokenizer.tokenize(temp)
        temp = temp.split(' ')
        count = 0
        temp_vect = self.np.zeros(50, dtype=float)
        for word in temp:
            if count < 300:
                if word in self._vocab:
                    temp_vect += self._w2v.wv[word]
                    count += 1
        temp_vect = temp_vect / float(count)
        return self._model.predict(self.np.array([temp_vect]))[0]