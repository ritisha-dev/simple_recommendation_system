import pandas as pd
import os

from src.feat_engg.feat_engg import *
from src.common.utils import *


class Features:

    def __init__(self, path):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path + "/preprocessed/preprocessed_data.csv")
        return df

    def transform(self, df):

        user_feat_df = user_feat(df)
        book_feat_df = book_feat(df)
        df = merge(df, user_feat_df, book_feat_df)
        df = rating_feat(df)
        item_sim_df = user_item_similarity(df)

        return df, item_sim_df

    def write(self, df, item_similarity_df):

        if not os.path.exists(self.path + "/features"):
            os.makedirs(self.path + "/features")

        df.to_csv(self.path + "/features/feat_data.csv", index=False)
        item_similarity_df.to_csv(
            self.path + "/features/item_similarity.csv", index=False
        )
        print(
            f"Features have been generated and saved to {self.path}/features/feat_data.csv"
        )
        print(
            f"User-Item similarity scores have been generated and saved to {self.path}features/item_similarity.csv"
        )

    def main(self):
        os.chdir(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        df = self.load()
        df, item_similarity_df = self.transform(df)
        self.write(df, item_similarity_df)
