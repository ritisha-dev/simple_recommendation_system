import pandas as pd


class Similarity_Score:

    def __init__(self, path, top_n=5):
        self.path = path
        self.top_n = top_n
        self.similarity_score = pd.read_csv(self.path + "/features/item_similarity.csv")
        self.user_feat_df = pd.read_csv(self.path + "/features/feat_data.csv")

    def predict(self, user_id):

        already_read = set(
            self.user_feat_df[
                (self.user_feat_df["user_id"] == user_id)
                & ~(self.user_feat_df["book_rating"].isnull())
            ]
            .sort_values("book_rating", ascending=False)["isbn"]
            .values
        )

        if len(already_read) == 0:
            print("Top rated books for you to get started:")
            recommendations = self.user_feat_df.sort_values(
                "book_rating_weighted", ascending=False
            )[
                [
                    "isbn",
                    "book_title",
                    "book_author",
                    "book_rating_mean",
                    "book_rating_count",
                    "book_rating_weighted",
                ]
            ].drop_duplicates()[
                : self.top_n + 1
            ]

        else:
            sim_df_tmp = self.similarity_score[
                (self.similarity_score["user_item_similarity"] != 1)
                & (self.similarity_score["isbn_self"].isin(already_read))
                & ~(self.similarity_score["isbn"].isin(already_read))
            ].sort_values("user_item_similarity", ascending=False)[
                ["isbn", "user_item_similarity"]
            ][
                : self.top_n + 1
            ]

            recommendations = (
                self.user_feat_df[
                    [
                        "isbn",
                        "book_title",
                        "book_author",
                        "book_rating_mean",
                        "book_rating_count",
                    ]
                ]
                .merge(sim_df_tmp, on="isbn", how="inner")
                .sort_values("user_item_similarity", ascending=False)
                .drop_duplicates()
            )
            print("Top recommendations based on your reading history:")

        return recommendations
