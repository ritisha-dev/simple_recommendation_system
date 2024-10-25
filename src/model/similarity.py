import pandas as pd
import numpy as np
import pickle


class Similarity_Score:

    def __init__(self, path, top_n=5):
        self.path = path
        self.top_n = top_n
        # self.similarity_score = pd.read_csv(self.path + "/features/item_similarity.csv")
        self.ratings_df = pd.read_csv(self.path + "/preprocessed/ratings_cleansed.csv")
        self.books_df = pd.read_csv(self.path + "/preprocessed/books_cleansed.csv")
        self.sim_mat = np.load(self.path + "/features/sim_mat.npy")
        with open(self.path + "/features/index_data_meta.pkl", "rb") as f:
            self.idx = pickle.load(f)

    def predict(self, df):

        recommendations_out = pd.DataFrame()

        idx_list = list(self.idx)
        user, isbn, rank = [], [], []
        self.ratings_df["read"] = (
            self.ratings_df.groupby("user_id")
            .apply(lambda x: list(x["isbn"].values))
            .reset_index(drop=True)
        )

        unique_users = df[df["user_rating_count"] >= 7]["user_id"].unique()
        bookisbn_top = list(
            df[["isbn", "user_rating_count"]]
            .drop_duplicates()
            .sort_values("user_rating_count", ascending=False)["isbn"]
        )[: self.top_n]

        for u_id in iter(unique_users):

            user_id = u_id

            already_read = [
                i
                for i in self.ratings_df[self.ratings_df["user_id"] == user_id][
                    "read"
                ].values
                if i in idx_list
            ]

            # already_read = [i for i in already_read if i in idx_list]

            if len(already_read) > 0:
                # print("Top rated books for you to get started:")
                idx_exclude = [idx_list.index(id) for id in already_read]
                idx_rec = [
                    i for i in range(self.sim_mat.shape[0]) if i not in idx_exclude
                ]
                all_isbns_idx = {}
                for i in already_read:
                    i_idx = idx_list.index(i)
                    idx_tmp = (
                        self.sim_mat[idx_rec, i_idx]
                        .argsort()[-self.top_n :][::-1]
                        .tolist()
                    )
                    all_isbns_idx = {j: self.sim_mat[i_idx, j] for j in idx_tmp}

                all_isbns_idx = dict(
                    sorted(
                        all_isbns_idx.items(), key=lambda item: item[1], reverse=True
                    )
                )
                bookisbn = [idx_list[i] for i in all_isbns_idx.keys()]

            else:
                bookisbn = bookisbn_top

            user.extend([str(user_id)] * len(bookisbn))
            isbn.extend(bookisbn)
            rank.extend(list(range(1, len(bookisbn) + 1)))

        recommendations_out = pd.DataFrame(
            {
                "user_id": user,
                "isbn": isbn,
                "rank": rank,
            }
        )

        recommendations_out = recommendations_out.merge(
            self.books_df, on="isbn", how="left"
        )
        return recommendations_out
