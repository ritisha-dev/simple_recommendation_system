import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def user_feat(df):
    user_feat_df = (
        df.groupby("user_id")
        .agg(
            user_rating_count=pd.NamedAgg(column="book_rating", aggfunc="count"),
            user_rating_mean=pd.NamedAgg(column="book_rating", aggfunc="mean"),
        )
        .reset_index()
    )
    return user_feat_df


def book_feat(df):

    min_max_scaler = MinMaxScaler()

    book_feat_df = (
        df.groupby("isbn")
        .agg(
            book_rating_count=pd.NamedAgg(column="user_id", aggfunc="nunique"),
            book_rating_mean=pd.NamedAgg(column="book_rating", aggfunc="mean"),
            book_rating_min=pd.NamedAgg(column="book_rating", aggfunc="min"),
            book_rating_max=pd.NamedAgg(
                column="book_rating",
                aggfunc="max",
            ),
        )
        .reset_index()
    )
    book_feat_df["book_rating_scaled"] = (
        book_feat_df["book_rating_mean"] - book_feat_df["book_rating_min"]
    ) / (book_feat_df["book_rating_max"] - book_feat_df["book_rating_min"])
    book_feat_df["book_rating_count_scaled"] = min_max_scaler.fit_transform(
        book_feat_df["book_rating_count"].values.reshape(-1, 1)
    )[:, 0]

    book_feat_df["book_rating_weighted"] = (
        book_feat_df["book_rating_scaled"] * book_feat_df["book_rating_count_scaled"]
    )

    return book_feat_df


def rating_feat(df):

    df["rating_dir"] = df["book_rating"].apply(
        lambda x: np.where(-1, x <= 3, np.where(3 < x <= 5, 0, 1))
    )

    return df


def user_item_similarity(df, user_rating_count_thr=7, book_rating_count_thr=10):

    filtered_df = df[
        (df["user_rating_count"] >= user_rating_count_thr)
        & (df["book_rating_count"] >= book_rating_count_thr)
    ]
    pivot_df = filtered_df.pivot(
        index="isbn", columns="user_id", values="book_rating"
    ).fillna(0)

    sc = StandardScaler()
    filtered_mat = sc.fit_transform(pivot_df)

    sim_mat = cosine_similarity(filtered_mat)
    sim_mat[np.arange(sim_mat.shape[0])[:, None] >= np.arange(sim_mat.shape[1])] = (
        np.nan
    )

    item_sim_df = pd.DataFrame(sim_mat, columns=pivot_df.index).reset_index(drop=True)
    item_sim_df["isbn_self"] = pivot_df.index

    item_sim_df = pd.melt(
        item_sim_df,
        id_vars="isbn_self",
        var_name="isbn",
        value_name="user_item_similarity",
    ).dropna()

    return item_sim_df
