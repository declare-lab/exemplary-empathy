import time
import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataloader import MainDataLoader
from nlgevaluation import compute_bleu
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from models import ERGModel, ERGMainModel

def configure_dataloaders(batch_size):
    "Prepare dataloaders"
    train_loader = MainDataLoader("data/empathetic_dialogues/valid_dpr.csv", batch_size, shuffle=True)
    valid_loader = MainDataLoader("data/empathetic_dialogues/valid_dpr.csv", batch_size, shuffle=False)
    test_loader = MainDataLoader("data/empathetic_dialogues/test_dpr.csv", batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def configure_optimizer(model, args):
    "Prepare optimizer"
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
        "lr": args.lr
    }
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

def configure_scheduler(optimizer, num_training_steps, args):
    "Prepare scheduler"
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler

def train_or_eval_model(model, dataloader, optimizer=None, train=False, main_loss_w=1, empathy_loss_w=1, sentiment_loss_w=1):
    losses = []
    assert not train or optimizer!=None

    if train:
        model.train()
    else:
        model.eval()
        
    for conv_id, emotion, context, response, exemplars, empathy1_labels, empathy2_labels, empathy3_labels, sentiment in tqdm(dataloader, leave=False):
    
        if train:
            optimizer.zero_grad()

        out, empathy1_preds, empathy2_preds, empathy3_preds, sentiment_preds = model(context, response, exemplars=exemplars)

        empathy1_labels = torch.tensor(empathy1_labels).cuda()
        empathy2_labels = torch.tensor(empathy2_labels).cuda()
        empathy3_labels = torch.tensor(empathy3_labels).cuda()
        sentiment_labels = torch.tensor(sentiment).cuda()

        loss = out.loss

        empathy1_loss = empathy_loss_function(empathy1_preds, empathy1_labels)
        empathy2_loss = empathy_loss_function(empathy2_preds, empathy2_labels)
        empathy3_loss = empathy_loss_function(empathy3_preds, empathy3_labels)
        total_empathy_loss = empathy1_loss + empathy2_loss + empathy3_loss

        sentiment_loss = sentiment_loss_function(sentiment_preds, sentiment_labels)
        
        total_loss = main_loss_w * loss + empathy_loss_w * total_empathy_loss + sentiment_loss_w * sentiment_loss

        if train:
            total_loss.backward()
            optimizer.step()

        # losses.append(total_loss.item())
        losses.append(loss.item()) # Return the generative seq2seq loss.

    avg_loss = round(np.mean(losses), 4)
    return avg_loss

def test_model(model, dataloader, mode):
    references, hypothesis, utt_ids = [], [], []
    for conv_id, emotion, context, response, exemplars, empathy1_labels, empathy2_labels, empathy3_labels, sentiment in tqdm(dataloader, leave=False):
    
        ref = [[item] for item in response]
        hyp = model.erg_model.generate(context, exemplars=exemplars, mode=mode)

        references += ref
        hypothesis += hyp
        utt_ids += [[conv, len(con) + 1, "\n".join(con)] for conv, con in zip(conv_id, context)]

    scores = compute_bleu(references, hypothesis)
    bleu1 = round(scores["Bleu_1"], 4)
    bleu2 = round(scores["Bleu_2"], 4)

    return bleu1, bleu2, references, hypothesis, utt_ids

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--adam-beta1", default=0.9, type=float, help="beta1 for AdamW optimizer.")
    parser.add_argument("--adam-beta2", default=0.999, type=float, help="beta2 for AdamW optimizer.")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Steps used for a linear warmup from 0 to lr.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Ratio of total training steps used for a linear warmup from 0 to lr.")
    parser.add_argument("--src-len", type=int, default=200, help="Max source length.")
    parser.add_argument("--tgt-len", type=int, default=50, help="Max target length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs.")
    parser.add_argument("--model", default="t5-small", help="Which seq2seq model.")
    parser.add_argument("--add-exemplars", action="store_true", default=True, help="Whether to use add exemplars.")
    parser.add_argument("--max-exemplars", type=int, default=10, help="Number of exemplars")
    parser.add_argument("--decode", default="topk", help="topk or beam search decoding strategy.")
    parser.add_argument("--inference", default=None, help="run_ID")
    parser.add_argument("--main-loss-w", type=float, default=1.0, help="loss weight for the generative loss")
    parser.add_argument("--empathy-loss-w", type=float, default=1.0, help="loss weight for the empathy loss")
    parser.add_argument("--sentiment-loss-w", type=float, default=1.0, help="loss weight for the sentiment loss")
    parser.add_argument("--strategy", type=int, default=2, help="logits to probability strategy for empathy/sentiment prediction model inputs")
    parser.add_argument("--fixed", action="store_true", default=False, help="Whether to keep empathy and sentiment prediction models fixed.")

    args = parser.parse_args()
    print(args)

    global max_source_length
    global max_target_length

    if args.inference is None:
        run_ID = int(time.time())
        print(f"run ID: {run_ID}")

    max_source_length, max_target_length = args.src_len, args.tgt_len
    batch_size = args.batch_size
    n_epochs = args.epochs
    model_name = args.model
    strategy = args.strategy
    main_loss_w = args.main_loss_w
    empathy_loss_w = args.empathy_loss_w
    sentiment_loss_w = args.sentiment_loss_w

    train_loader, valid_loader, test_loader = configure_dataloaders(batch_size)

    empathy_loss_function = torch.nn.CrossEntropyLoss().cuda()
    sentiment_loss_function = torch.nn.MSELoss().cuda()

    if args.inference is None:
        model = ERGMainModel(model_name, max_source_length, max_target_length, strategy, args.add_exemplars, args.max_exemplars, args.fixed).cuda()
        print ("Num trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = configure_optimizer(model, args)

        best_loss = None
        for e in range(n_epochs):
            train_loss = train_or_eval_model(model, train_loader, optimizer, True, main_loss_w=main_loss_w, empathy_loss_w=empathy_loss_w, sentiment_loss_w=sentiment_loss_w)
            valid_loss = train_or_eval_model(model, valid_loader, main_loss_w=main_loss_w, empathy_loss_w=empathy_loss_w, sentiment_loss_w=sentiment_loss_w)
            print ("Epoch {}: train loss: {}, valid loss: {}".format(e+1, train_loss, valid_loss))

            if best_loss == None or best_loss > valid_loss:
                if not os.path.isdir(f"saved/{run_ID}"):
                    os.mkdir(f"saved/{run_ID}")
                torch.save(model.state_dict(), f"saved/{run_ID}/model.pt")
                best_loss = valid_loss

    else:
        run_ID = args.inference
        
    model = ERGMainModel(model_name, max_source_length, max_target_length, strategy, args.add_exemplars, args.max_exemplars).cuda()
    model.load_state_dict(torch.load(f"saved/{run_ID}/model.pt"))
    model.eval()

    test_loss = train_or_eval_model(model, test_loader, main_loss_w=main_loss_w, empathy_loss_w=empathy_loss_w, sentiment_loss_w=sentiment_loss_w)
    ppl = np.exp(test_loss)
    print(f"Perplexity {ppl}")

    bleu1, bleu2, references, hypothesis, utt_ids = test_model(model, test_loader, mode=args.decode)

    if "best_loss" in locals():
        content = [str(bleu1), str(bleu2), str(round(best_loss, 6)), str(round(ppl, 6)), str(run_ID), str(args)]
    else:
        content = [str(bleu1), str(bleu2), "loss not available", "perplexity not available", str(run_ID), str(args)]

    with open("results/results.txt", "a") as f:
        f.write("\t".join(content) + "\n")

    with open(f"output/{run_ID}_output.tsv", "w") as f:
        pd.DataFrame({
            "conv_id": [x[0] for x in utt_ids],
            "utterance_idx": [x[1] for x in utt_ids],
            "context": [x[2] for x in utt_ids],
            "reference": [x[0] for x in references],
            "generated": hypothesis
            }).to_csv(f, sep="\t")
        
    pickle.dump(references, open("output/"+str(run_ID)+"_references.pkl", "wb"))
    pickle.dump(hypothesis, open("output/"+str(run_ID)+"_hypothesis.pkl", "wb"))

