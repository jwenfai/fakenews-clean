from pathlib import Path
import pandas as pd

volkova_dir = Path("K:/Work/Datasets-FakeNews/source-reliability/fakenews_dataset/paper_data")

hydrated_path = volkova_dir / "multiclass_tweets_hydrated.csv"
lbl_path = volkova_dir / "multiclass_tweets.csv"
twt_df_path = volkova_dir / "volkova_fake_satire.csv"
twt_id_path = volkova_dir / "volkova_rehydrated_id.csv"

hydrated_dtypes = {"in_reply_to_status_id": "Int64", "in_reply_to_user_id": "Int64", "reweet_id": "Int64"}

hydrated_df = pd.read_csv(hydrated_path, dtype=hydrated_dtypes)
hydrated_df.rename(columns={'reweet_id': 'retweet_id'}, inplace=True)
hydrated_df.drop(columns=["user_screen_name.1"], inplace=True)
lbl_df = pd.read_csv(lbl_path)

twt_df = pd.merge(hydrated_df, lbl_df[["tweet_id", "label"]], left_on="id", right_on="tweet_id")
twt_df[["tweet_id"]].to_csv(twt_id_path, sep="\t", index=False)
twt_df.drop(columns=["tweet_id"], inplace=True)
twt_df.to_csv(twt_df_path, sep="\t", index=False)

# Dataset statistics
import numpy as np
from nltk import sent_tokenize
twt_df = pd.read_csv(twt_df_path, sep="\t")
n_sents = [len(sent_tokenize(row)) for row in twt_df.text]
print(np.mean(n_sents), np.median(n_sents), np.max(n_sents))

# Number of domain names in dataset
from urllib.parse import urlparse
domain_names = [urlparse(x).netloc for x in twt_df.urls if type(x) == str]
print(len(np.unique(domain_names)))
