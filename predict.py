import pandas as pd
import pickle
from data_prep import DataPrep
from model import Model
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
from open_psychometrics import Big5
import scipy.stats as stats
from math import pi
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import datetime

class Predictor():
    def __init__(self):
        self.traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
        self.models = {}
        self.load_models()
        self.df = self.load_df()
        # self.df = self.agg_avg_personality()

    def load_models(self):
        m = Model()
        for trait in self.traits:
            with open('static/' + trait + '_model.pkl', 'rb') as f:
                self.models[trait] = pickle.load(f)

    def load_df(self):
        df = "hi how are you !"
        return df


    def predict(self, X, traits='All', predictions='All'):
        predictions = {}
        if traits == 'All':
            for trait in self.traits:
                pkl_model = self.models[trait]

                
                trait_scores = pkl_model.predict(X, regression=True).reshape(1, -1)
                # scaler = MinMaxScaler(feature_range=(0, 50))
                # print(scaler.fit_transform(trait_scores))
                # scaled_trait_scores = scaler.fit_transform(trait_scores)
                predictions['pred_s'+trait] = trait_scores.flatten()[0]
                # predictions['pred_s'+trait] = scaled_trait_scores.flatten()

                trait_categories = pkl_model.predict(X, regression=False)
                predictions['pred_c'+trait] = str(trait_categories[0])
                # predictions['pred_c'+trait] = trait_categories

                trait_categories_probs = pkl_model.predict_proba(X)
                predictions['pred_prob_c'+trait] = trait_categories_probs[:, 1][0]
                # predictions['pred_prob_c'+trait] = trait_categories_probs[:, 1]

        print(predictions)
        return predictions


    def agg_avg_personality(self):

        df = "laskdjf"
        # df_mean_scores = df.groupby('NAME')[[
        #     'pred_sOPN', 'pred_sCON', 'pred_sEXT', 'pred_sAGR', 'pred_sNEU',
        # ]].mean()

        # df_mean_scores = self.df.groupby(['NAME'], as_index=False).agg(
        #                       {'pred_sOPN':['mean'], 'pred_sCON':['mean'], 'pred_sEXT':['mean'], 'pred_sAGR':['mean'], 'pred_sNEU':['mean']})
        #
        # df_mean_scores.columns = ['NAME', 'avg_pred_sOPN', 'avg_pred_sCON', 'avg_pred_sEXT', 'avg_pred_sAGR', 'avg_pred_sNEU']
        #
        # df = self.df.merge(df_mean_scores, how='right', on='NAME')
        #
        # # df_mean_scores = df.groupby('NAME')[[
        # #     'pred_prob_cOPN', 'pred_prob_cCON', 'pred_prob_cEXT', 'pred_prob_cAGR', 'pred_prob_cNEU'
        # # ]].mean()
        #
        # df_mean_probs = df.groupby(['NAME'], as_index=False).agg(
        #                       {'pred_prob_cOPN':['mean'], 'pred_prob_cCON':['mean'], 'pred_prob_cEXT':['mean'], 'pred_prob_cAGR':['mean'], 'pred_prob_cNEU':['mean']})
        # df_mean_probs.columns = ['NAME', 'avg_pred_prob_cOPN', 'avg_pred_prob_cCON', 'avg_pred_prob_cEXT', 'avg_pred_prob_cAGR', 'avg_pred_prob_cNEU']
        #
        # df = df.merge(df_mean_probs, how='right', on='NAME')

        return df




if __name__ == '__main__':
    p = Predictor()
    # P.predict("Hi iam a baboon")
    df = pd.read_csv('./data/myPersonality/test.csv', encoding="ISO-8859-1")

    df = df["STATUS"]
    len(df)

    opn = []
    con = []
    ext = []
    agr = []
    neu = []
    for item in df:
        predictions = p.predict([item])
        traits = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']
        for trait in traits:
            print(predictions['pred_'+ trait ])
            if(trait == 'cOPN'):
                opn.append(predictions["pred_cOPN"])
            if (trait == 'cCON'):
                con.append(predictions["pred_cCON"])

            if (trait == 'cEXT'):
                ext.append(predictions["pred_cEXT"])

            if (trait == 'cNEU'):
                neu.append(predictions["pred_cNEU"])

            if (trait == 'cAGR'):
                agr.append(predictions["pred_cAGR"])

    DF = pd.DataFrame(list(zip(opn,con,ext,agr,neu)),columns=traits)
    DF.to_csv("output.csv")


            # DF.to_csv("./data/myPersonality/output.csv")


