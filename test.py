import numpy as np
import pandas as pd
from sklearn import metrics


df1 = pd.read_csv("./output.csv")
df2 = pd.read_csv("./data/myPersonality/test.csv")

df1 = df1[['cOPN','cCON','cEXT','cAGR','cNEU']]
df2 = df2[['cOPN','cCON','cEXT','cAGR','cNEU']]

def convert_traits_to_boolean(df):
    trait_columns = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']
    d = {'y': True, 'n': False}

    for trait in trait_columns:
        df[trait] = df[trait].map(d)

    return df

df2 = convert_traits_to_boolean(df2)

traits = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']
for trait in traits:
    print("Confusion Matrix for: " + trait)
    print(metrics.confusion_matrix(df1[[trait]], df2[[trait]]))
    print("\n")

    print(metrics.classification_report(df1[[trait]], df2[[trait]]))
    print("\n")

    print("acuracy = " + str(metrics.accuracy_score(df1[[trait]], df2[[trait]])))
    print("\n")

    print("----------------------------------------------------------")
    print("\n")
