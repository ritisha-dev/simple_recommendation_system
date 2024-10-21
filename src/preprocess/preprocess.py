from src.common.utils import *
import pandas as pd

import os


class Preprocess:
    def __init__(self, path):
        self.path = path

    def load(self):

        df_users = pd.read_csv(self.path + "/raw_data/Users.csv")
        df_ratings = pd.read_csv(self.path + "/raw_data/Ratings.csv")
        df_books = pd.read_csv(self.path + "/raw_data/Books.csv")
        return df_users, df_ratings, df_books

    def transform(self, df_users, df_ratings, df_books):
        df_users, df_ratings, df_books = (
            clean_colname(df_users),
            clean_colname(df_ratings),
            clean_colname(df_books),
        )

        df_books = fix_dtype(df_books)

        df_users, df_ratings, df_books = (
            apply_whitespace(df_users),
            apply_whitespace(df_ratings),
            apply_whitespace(df_books),
        )

        df_users, df_ratings, df_books = (
            df_users.drop_duplicates(),
            df_ratings.drop_duplicates(),
            df_books.drop_duplicates(),
        )

        df = merge(df_users, df_ratings, df_books)

        return df, df_users, df_ratings, df_books

    def write(self, df, df_users, df_ratings, df_books):

        if not os.path.exists(self.path + "/preprocessed"):
            os.makedirs(self.path + "/preprocessed")

        df_books.to_csv(self.path + "/preprocessed/books_cleansed.csv", index=False)
        df_ratings.to_csv(self.path + "/preprocessed/ratings_cleansed.csv", index=False)
        df_users.to_csv(self.path + "/preprocessed/users_cleansed.csv", index=False)
        df.to_csv(self.path + "/preprocessed/preprocessed_data.csv", index=False)
        print(
            f"Data has been preprocessed and saved to {self.path}/preprocessed/preprocessed_data.csv"
        )

    def main(self):
        os.chdir(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        df_users, df_ratings, df_books = self.load()
        df, df_users, df_ratings, df_books = self.transform(
            df_users, df_ratings, df_books
        )
        self.write(df, df_users, df_ratings, df_books)
