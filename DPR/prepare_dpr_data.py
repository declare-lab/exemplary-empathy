import json
import random
import pandas as pd, numpy as np
from tqdm import tqdm

def preprocess(text):
    return text.replace("_comma_", ",")

def load_empathetic_dialogues(split):
    data = pd.read_csv("../data/empathetic_dialogues/original/" + split + ".csv", quoting=3).drop(columns=["junk"])
    data["utterance"] = data["utterance"].apply(lambda x: preprocess(x))
    history = []
    conv_ids = list(dict.fromkeys(data["conv_id"]))
    for ids in tqdm(conv_ids):
        conv_utts = data[data["conv_id"]==ids]["utterance"]
        history += [" ".join(conv_utts[:i]) for i in range(len(conv_utts))]
    data["history"] = history
    return data

def prepare_dpr_empathetic_dialogues(query_df):
    dpr_data = []
    for j in tqdm(range(len(query_df))):
        row = query_df.iloc[j]
        
        if row["utterance_idx"]%2 == 0:
            query = row["history"]
            emotion, conv_id = row["context"], row["conv_id"]
            positive_instance = row["utterance"]

            neg_candidates = list(query_df[
                (query_df["context"] != emotion)
                & (query_df["conv_id"] != conv_id)
                & (query_df["utterance_idx"]%2 == 0)
            ].index)

            hard_neg_candidates = list(query_df[
                (query_df["context"] == emotion)
                & (query_df["conv_id"] != conv_id)
                & (query_df["utterance_idx"]%2 == 0)
            ].index)

            neg_instance = query_df.iloc[random.choice(neg_candidates)]["utterance"]
            hard_neg_instance = query_df.iloc[random.choice(hard_neg_candidates)]["utterance"]

            json_instance = {
                "question": query,
                "answers": [""],
                "positive_ctxs": [{"title": "", "text": positive_instance}],
                "negative_ctxs": [{"title": "", "text": neg_instance}],
                "hard_negative_ctxs": [{"title": "", "text": hard_neg_instance}]
            }
            dpr_data.append(json_instance)
            
    return dpr_data

def prepare_dpr_reddit(query_df):
    dpr_data = []
    for j in tqdm(range(len(query_df))):
        row = query_df.iloc[j]
        
        if row["label"] == 1:
            query = row["seeker_post"]
            positive_instance = row["response_post"]
            sp_id = row["sp_id"]

            neg_candidates = list(query_df[
                (query_df["label"] == 0)
                & (query_df["index"] != j)
                & (query_df["sp_id"] != sp_id)
            ].index)

            hard_neg_candidates = list(query_df[
                (query_df["label"] == 1)
                & (query_df["index"] != j)
                & (query_df["sp_id"] != sp_id)
            ].index)

            neg_instance = query_df.iloc[random.choice(neg_candidates)]["response_post"]
            hard_neg_instance = query_df.iloc[random.choice(hard_neg_candidates)]["response_post"]

            json_instance = {
                "question": query,
                "answers": [""],
                "positive_ctxs": [{"title": "", "text": positive_instance}],
                "negative_ctxs": [{"title": "", "text": neg_instance}],
                "hard_negative_ctxs": [{"title": "", "text": hard_neg_instance}]
            }
            dpr_data.append(json_instance)
            
    return dpr_data


## Emapthetic Dialogues ##
train = load_empathetic_dialogues("train")
valid = load_empathetic_dialogues("valid")

train_dpr = prepare_dpr_empathetic_dialogues(train)
valid_dpr = prepare_dpr_empathetic_dialogues(valid)

with open("dpr/new_data/empd-train.json", "w") as f:
    json.dump(train_dpr, f, indent=4, sort_keys=True)
with open("dpr/new_data/empd-valid.json", "w") as f:
    json.dump(valid_dpr, f, indent=4, sort_keys=True)
    
## Reddit Portion of Empathy Mental Health ##
train_reddit = pd.read_csv("../data/empathy_mental_health/emotion_train.csv")
train_reddit["index"] = range(len(train_reddit))

valid_reddit = pd.read_csv("../data/empathy_mental_health/emotion_valid.csv")
valid_reddit["index"] = range(len(valid_reddit))

train_reddit_dpr = prepare_dpr_reddit(train_reddit)
valid_reddit_dpr = prepare_dpr_reddit(valid_reddit)

with open("dpr/new_data/reddit-train.json", "w") as f:
    json.dump(train_reddit_dpr, f, indent=4, sort_keys=True)
with open("dpr/new_data/reddit-valid.json", "w") as f:
    json.dump(valid_reddit_dpr, f, indent=4, sort_keys=True)

## Combined ##
with open("dpr/new_data/empd-reddit-train.json", "w") as f:
    json.dump(train_dpr+train_reddit_dpr, f, indent=4, sort_keys=True)
with open("dpr/new_data/empd-reddit-valid.json", "w") as f:
    json.dump(valid_dpr+valid_reddit_dpr, f, indent=4, sort_keys=True)