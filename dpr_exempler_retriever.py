from tqdm import tqdm
import torch
import numpy as np, pandas as pd
import faiss.contrib.torch_utils
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

def preprocess(text):
    return text.replace("_comma_", ",")

def load_df(split):
    data = pd.read_csv("github/data/empathetic_dialogues/original/" + split + ".csv", quoting=3).drop(columns=["junk"])
    data["utterance"] = data["utterance"].apply(lambda x: preprocess(x))
    history = []
    conv_ids = list(dict.fromkeys(data["conv_id"]))
    for ids in tqdm(conv_ids):
        conv_utts = data[data["conv_id"]==ids]["utterance"]
        history += [" ".join(conv_utts[:i]) for i in range(len(conv_utts))]
    data["history"] = history
    return data

def embeddings_from_sentences(tokenizer, model, sentences):
    batch_size = 32
    embeddings = []
    for j in tqdm(range(0, len(sentences), batch_size)):
        batch = tokenizer(sentences[j:j+batch_size], padding=True, max_length=512, return_tensors="pt")
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        with torch.no_grad():
            output = model(input_ids, attention_mask)
        embeddings.append(output.pooler_output)
    return torch.cat(embeddings)

def compute_exemplers(query_df, query_tensor):
    k = 2048
    D, I = gpu_index_flat.search(query_tensor, k)
    exemplers = list(train["utterance"])
    df_exemplers, df_exemplers_indices = [], []

    for j in tqdm(range(len(query_df))):
        row = query_df.iloc[j]
        emotion, conv_id = row["context"], row["conv_id"]
        candidate_indices = set(train[
            (train["context"] == emotion) # dialogs having same emotion
            & (train["conv_id"] != conv_id) # different dialog (only required when computing train set exemplers)
            & (train["utterance_idx"]%2 == 0) # user 2 utterances
        ].index)

        retrieved, matches = I[j].cpu().numpy(), []
        for item in retrieved:
            if item in candidate_indices:
                matches.append(item)
            if len(matches) == 10:
                break

        df_exemplers_indices.append(" ææ ".join([str(ind) for ind in matches]))
        df_exemplers.append(" ææ ".join([exemplers[ind] for ind in matches]))
        
    query_df["exemplers"] = df_exemplers
    query_df["exemplers_index"] = df_exemplers_indices
    return query_df


if __name__ == "__main__":

    global train
    global gpu_index_flat

    # Load data in pandas
    test = load_df("test")
    valid = load_df("valid")
    train = load_df("train")

    # Load DPR models
    print ("Loading Models.")
    ctx_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
    qs_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda()
    ctx_model.eval(); qs_model.eval()

    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    qs_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    # Compute DPR encodings: train utterances as context; and train, val, test dialogue history as query
    print ("Computing embeddings.")
    test_query = embeddings_from_sentences(qs_tokenizer, qs_model, list(test["history"]))
    valid_query = embeddings_from_sentences(qs_tokenizer, qs_model, list(valid["history"]))
    train_query = embeddings_from_sentences(qs_tokenizer, qs_model, list(train["history"]))
    train_ctx = embeddings_from_sentences(ctx_tokenizer, ctx_model, list(train["utterance"]))


    # Indexing and maximum inner product search with faiss
    dim = train_ctx.size(1)
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(dim)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(train_ctx)

    # Retrieve exemplers
    print ("Retrieving exemplers.")
    test_dpr = compute_exemplers(test, test_query)
    valid_dpr = compute_exemplers(valid, valid_query)
    train_dpr = compute_exemplers(train, train_query)

    # Save
    test_dpr.to_csv("github/data/empathetic_dialogues/original/test_dpr.csv")
    valid_dpr.to_csv("github/data/empathetic_dialogues/original/valid_dpr.csv")
    train_dpr.to_csv("github/data/empathetic_dialogues/original/train_dpr.csv")
    print ("Done.")

