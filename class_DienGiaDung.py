class Filter_DienGiaDung():
    from pyvi import ViTokenizer  # thư viện NLP tiếng Việt
    import numpy as np
    import gensim  # thư viện NLP

    from sklearn.decomposition import TruncatedSVD
    from flask import Flask
    from flask import jsonify
    import pickle
    # tfidf thường

    def init(self,link_save_tfidf, link_save_X_data_tfidf, link_save_mode_xgb_2):
        path = open(

            link_save_tfidf, 'rb')
        self.tfidf_vect = self.pickle.load(path)

        # tfidf_svd
        self.svd = self.TruncatedSVD(n_components=300, random_state=42)
        path = open(
            link_save_X_data_tfidf, 'rb')
        self.X_data_tfidf = self.pickle.load(path)
        self.svd.fit(self.X_data_tfidf)

        # model
        path = open(
            link_save_mode_xgb_2, 'rb')
        self.bst = self.pickle.load(path)

    def xuLy(self, X_data_all):
        X_data_ok = self.np.array(X_data_all)
        X_data_ok.astype('str')
        X_data = []

        for line in X_data_ok:
            line = self.gensim.utils.simple_preprocess(line)
            line_1 = ' '.join(line)
            line_2 = self.ViTokenizer.tokenize(line_1)
            X_data.append(line_2)
        return X_data

    def classify(self, content):
        X_test = [content]
        X_test = self.xuLy(X_test)
        # global  X_test_tfidf, X_test_tfidf_svd
        # bst = xgb.XGBClassifier()
        # bst = bst.load_model('save_model_xgb.json')
        X_test_tfidf = self.tfidf_vect.transform(X_test)
        X_test_tfidf_svd = self.svd.transform(X_test_tfidf)
        test_predictions = self.bst.predict(X_test_tfidf_svd)
        # print(test_predictions)
        return test_predictions


if __name__ == '__main__':
    
    link_save_tfidf = '/home/xa/Documents/XApython/codeVcc/classification/save_tfidf.pkl'
    link_save_X_data_tfidf = '/home/xa/Documents/XApython/codeVcc/classification/save_X_data_tfidf.pkl'
    link_save_mode_xgb_2 = "/home/xa/Documents/XApython/codeVcc/classification/save_mode_xgb_2.pkl"
    xa = Filter_DienGiaDung()
    xa.init(link_save_tfidf, link_save_X_data_tfidf, link_save_mode_xgb_2)
    for i in range(100):
        print(xa.classify("askdjfhaskjdfhasjk jkashdjkashk"))