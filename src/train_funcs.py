import os
import math
import torch
import torch.nn as nn
from .misc import save_as_pickle, load_pickle
import logging
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = args.model_save_path
    checkpoint_path = os.path.join(base_path,"task_train_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"task_val_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_f1, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_f1

def load_results(model_no=0, path="./trained_model/"):
    """ Loads saved results if exists """
    losses_path = os.path.join(path, "task_train_losses_per_epoch_%d.pkl" % model_no)
    accuracy_path = os.path.join(path, "task_train_f1_per_epoch_%d.pkl" % model_no)
    f1_path = os.path.join(path, "task_val_f1_per_epoch_%d.pkl" % model_no)
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path) and os.path.isfile(f1_path):
        losses_per_epoch = load_pickle("task_train_losses_per_epoch_%d.pkl" % model_no, path)
        accuracy_per_epoch = load_pickle("task_train_f1_per_epoch_%d.pkl" % model_no, path)
        f1_per_epoch = load_pickle("task_val_f1_per_epoch_%d.pkl" % model_no, path)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch, f1_per_epoch = [], [], []
    return losses_per_epoch, accuracy_per_epoch, f1_per_epoch

def evaluate_(output, labels, ignore_idx):
    ### ignore index 0 (padding) when calculating accuracy
    idxs = (labels != ignore_idx).squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]
    l = labels.squeeze()[idxs]; o = o_labels[idxs]

    if len(idxs) > 1:
        acc = (l == o).sum().item()/len(idxs)
    else:
        acc = (l == o).sum().item()
    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

    return acc, (o, l), f1_score(l, o)

def evaluate_results(net, data_loader, pad_id, cuda):
    logger.info("Evaluating test samples...")
    acc = 0; out_labels = []; true_labels = []
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                          e1_e2_start=e1_e2_start)
            
            accuracy, (o, l), _ = evaluate_(classification_logits, labels, ignore_idx=-1)
            out_labels += o; true_labels+= l
            acc += accuracy
    
    accuracy = acc/(i + 1)
    results = {
        "accuracy": accuracy,
        "precision": precision_score(true_labels, out_labels),
        "recall": recall_score(true_labels, out_labels),
        "f1": f1_score(true_labels, out_labels),
        "true_labels": true_labels,
        "out_labels": out_labels
    }
    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    
    return results
    