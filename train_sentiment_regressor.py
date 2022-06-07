import os
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from models import T5EncoderRegressor
from dataloader import RegressionLoader
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from sklearn.metrics import f1_score, accuracy_score

def configure_dataloaders(batch_size):
    "Prepare dataloaders"
    train_loader = RegressionLoader("data/empathetic_dialogues/train_vader.csv", batch_size, shuffle=True)
    valid_loader = RegressionLoader("data/empathetic_dialogues/valid_vader.csv", batch_size, shuffle=False)
    test_loader = RegressionLoader("data/empathetic_dialogues/test_vader.csv", batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def configure_transformer_optimizer(model, args):
    "Prepare AdamW optimizer for transformer encoders"
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.wd,
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


def train_or_eval_model(model, dataloader, optimizer=None, train=False):
    losses = []
    assert not train or optimizer!=None
    
    if train:
        model.train()
    else:
        model.eval()
    
    for utterances, gold_scores in tqdm(dataloader, leave=False):
        if train:
            optimizer.zero_grad()
                
        scores = model(utterances)       
        loss = loss_function(scores, torch.tensor(gold_scores).cuda())
             
        if train:
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())
        
    avg_loss = round(np.mean(losses), 4)
    return avg_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for transformers.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--adam-beta1", default=0.9, type=float, help="beta1 for AdamW optimizer.")
    parser.add_argument("--adam-beta2", default=0.999, type=float, help="beta2 for AdamW optimizer.")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Steps used for a linear warmup from 0 to lr.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Ratio of total training steps used for a linear warmup from 0 to lr.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs.")
    parser.add_argument("--size", default="base", help="Which model size for T5: base or large")
    
    args = parser.parse_args()
    print(args)
    
    global loss_function
    global tokenizer
    
    batch_size = args.batch_size
    n_epochs = args.epochs
    size = args.size
       
    run_ID = int(time.time())
    print ("run id:", run_ID)
    
    model = T5EncoderRegressor(size).cuda()
    loss_function = torch.nn.MSELoss().cuda()
    optimizer = configure_transformer_optimizer(model, args)
    
    train_loader, valid_loader, test_loader = configure_dataloaders(batch_size)
    
    lf = open("results/logs.tsv", "a")
    lf.write(str(run_ID) + "\tvader sentiment\t"  + str(args) + "\n")
    
    best_loss = np.inf
    for e in range(n_epochs):
        train_loss = train_or_eval_model(model, train_loader, optimizer, True)  
        valid_loss = train_or_eval_model(model, valid_loader)
        test_loss = train_or_eval_model(model, test_loader)
        
        x = "Epoch {}: train loss: {}; valid loss: {}; test loss: {}".format(e+1, train_loss, valid_loss, test_loss)
        print (x)
        lf.write(x + "\n")
        
        if valid_loss < best_loss:
            if not os.path.exists("saved/sentiment/" + str(run_ID) + "/"):
                os.makedirs("saved/sentiment/" + str(run_ID) + "/")
            torch.save(model.state_dict(), "saved/sentiment/"+ str(run_ID) + "/model.pt")
            best_loss = valid_loss
            
    lf.write("\n\n")
    lf.close()
    
    print ("Best valid loss: {}".format(best_loss))
    
    content = [str(best_loss), "vader sentiment", str(run_ID), str(args)]
    with open("results/results.txt", "a") as f:
        f.write("\t".join(content) + "\n")
        
   