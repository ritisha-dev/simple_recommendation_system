def clean_colname(df):
    df.columns = [c.lower().replace("-", "_") for c in df.columns]
    return df


def fix_dtype(df):
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df


def rm_whitespace(x):
    return x.strip()


def apply_whitespace(df):
    col_list = [c for c in df.columns if df[c].dtype == "object"]
    for c in col_list:
        df[c] = df[c].apply(rm_whitespace)
    return df


def merge(df_users, df_ratings, df_books):
    df = df_users.merge(df_ratings, on="user_id", how="left").merge(
        df_books, on="isbn", how="left"
    )
    return df
