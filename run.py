import argparse
import pandas as pd
import numpy as np

from src.common.utils import *
from src.preprocess.preprocess import Preprocess
from src.feat_engg.feat import Features
from src.model.similarity import Similarity_Score


def main():

    parser = argparse.ArgumentParser(description="Run the recommendation system.")
    parser.add_argument(
        "--path", type=str, default="data", help="Path to the data directory"
    )
    parser.add_argument("--training", default="False", type=str, help="Training flag")
    parser.add_argument(
        "--user_id", type=str, help="User ID for generating recommendations"
    )

    args = parser.parse_args()

    user_id = args.user_id

    try:
        path = args.path
        training = args.training
        user_id = args.user_id
        if path is None or training is None or user_id is None:
            raise ValueError(
                "path training and user_id are required for generating recommendations."
            )
    except:
        pass

    if training == "True":
        print("Training the model")
        # Prepare the data
        pp = Preprocess(path)
        pp.main()

        ft = Features(path)
        ft.main()

    else:
        print("Scoring data")
        sim = Similarity_Score(path=path)
        try:
            recommendations = sim.predict(user_id=user_id)
            print(recommendations)
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
