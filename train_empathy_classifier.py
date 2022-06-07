import os
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from models import T5EncoderClassifier
from dataloader import ClassificationLoader
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from sklearn.metrics import f1_score, accuracy_score

import transformers
transformers.logging.set_verbosity_error()

def configure_dataloaders(dimension, batch_size):
    "Prepare dataloaders"
    train_loader = ClassificationLoader("data/empathy_mental_health/" + dimension + "_train.csv", batch_size, shuffle=True)
    valid_loader = ClassificationLoader("data/empathy_mental_health/" + dimension + "_valid.csv", batch_size, shuffle=False)
    return train_loader, valid_loader

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
    losses, preds, labels = [], [], []
    assert not train or optimizer!=None
    
    if train:
        model.train()
    else:
        model.eval()
    
    for context, response, emp_labels in tqdm(dataloader, leave=False):
        if train:
            optimizer.zero_grad()
                
        logits = model(context, response)        
        loss = loss_function(logits, torch.tensor(emp_labels).cuda())
             
        if train:
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())
        preds.append(torch.argmax(logits, 1).data.cpu().numpy())
        labels.append(np.array(emp_labels))
        
    avg_loss = round(np.mean(losses), 4)
    
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)    
    accuracy = round(accuracy_score(labels, preds)*100, 2)
    fscore = round(f1_score(labels, preds, average="weighted")*100, 2)
    
    return avg_loss, accuracy, fscore

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for transformers.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--adam-beta1", default=0.9, type=float, help="beta1 for AdamW optimizer.")
    parser.add_argument("--adam-beta2", default=0.999, type=float, help="beta2 for AdamW optimizer.")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Steps used for a linear warmup from 0 to lr.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Ratio of total training steps used for a linear warmup from 0 to lr.")
    parser.add_argument("--dim", default="emo", help="Which empathetic dimension")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs.")
    parser.add_argument("--size", default="base", help="Which model size for T5: base or large")
    
    args = parser.parse_args()
    print(args)
    
    global loss_function
    global tokenizer
    
    if args.dim == "emo":
        dimension = "emotion"
    elif args.dim == "exp":
        dimension = "exploration"
    elif args.dim == "int":
        dimension = "interpretation"
    
    batch_size = args.batch_size
    n_epochs = args.epochs
    size = args.size
       
    run_ID = int(time.time())
    print ("run id:", run_ID)
    
    model = T5EncoderClassifier(size).cuda()
    loss_function = torch.nn.CrossEntropyLoss().cuda()
    optimizer = configure_transformer_optimizer(model, args)
    
    train_loader, valid_loader = configure_dataloaders(dimension, batch_size)
    
    lf = open("results/logs.tsv", "a")
    lf.write(str(run_ID) + "\tempathy " + dimension + "\t" + str(args) + "\n")
    
    best_score, best_acc = 0, 0
    for e in range(n_epochs):
        train_loss, train_acc, train_fscore = train_or_eval_model(model, train_loader, optimizer, True)  
        valid_loss, valid_acc, valid_fscore = train_or_eval_model(model, valid_loader)
        
        x = "Epoch {}: train loss: {}, acc: {}, fscore: {}; valid loss: {}, acc: {}, fscore: {}".format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore)
        print (x)
        lf.write(x + "\n")
        
        if best_score < valid_fscore:
            if not os.path.exists("saved/empathy/" + str(run_ID) + "/"):
                os.makedirs("saved/empathy/" + str(run_ID) + "/")
            torch.save(model.state_dict(), "saved/empathy/"+ str(run_ID) + "/model.pt")
            best_score, best_acc = valid_fscore, valid_acc
            
    lf.write("\n\n")
    lf.close()
    
    print ("Best valid acc: {}, fscore: {}".format(valid_acc, valid_fscore))
    
    content = [str(valid_acc), str(valid_fscore), "empathy " + dimension, str(run_ID), str(args)]
    with open("results/results.txt", "a") as f:
        f.write("\t".join(content) + "\n")
        
   