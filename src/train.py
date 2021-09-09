import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .train_funcs import load_state, load_results, evaluate_, evaluate_results
from .misc import save_as_pickle
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_and_fit(args, net, train_loader, val_loader, tokenizer):

    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
        
    logger.info("FREEZING MOST HIDDEN LAYERS...")
    if args.model_no == 0:
        unfrozen_layers = ["classifier", "pooler", \
                           #"encoder.layer.4", "encoder.layer.5", "encoder.layer.6", "encoder.layer.7", \
                           "encoder.layer.8", "encoder.layer.9", "encoder.layer.10","encoder.layer.11", \
                           "classification_layer", "blanks_linear", "lm_linear", "cls"]
    elif args.model_no == 1:
        unfrozen_layers = ["classifier", "pooler", "classification_layer",\
                           "blanks_linear", "lm_linear", "cls",\
                           "albert_layer_groups.0.albert_layers.0.attention", \
                           "albert_layer_groups.0.albert_layers.0.ffn"]#, \
                           #"albert_layer_groups.0.albert_layers.0.full_layer_layer_norm"]
        
    for name, param in net.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            #print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            #print("[FREE]: %s" % name)
            param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                      24,26,30], gamma=0.8)
    
    start_epoch, best_f1 = 0, 0 #load_state(net, optimizer, scheduler, args, load_best=False)  

    losses_per_epoch, f1_per_epoch, val_f1_per_epoch = [], [], [] #load_results(args.model_no)
    
    logger.info("Starting training process...")
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    update_size = len(train_loader)//10
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        net.train(); total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; total_f1 = 0.0; accuracy_per_batch = []; f1_per_batch = []
        for i, data in enumerate(train_loader, 0):
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
            
            loss = criterion(classification_logits, labels.squeeze(1))
            loss = loss/args.gradient_acc_steps
            
            loss.backward()
            
            grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
            
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            result = evaluate_(classification_logits, labels, ignore_idx=-1)
            total_acc += result[0]
            total_f1 += result[-1]
            
            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                accuracy_per_batch.append(total_acc/update_size)
                f1_per_batch.append(total_f1/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss, accuracy, f1 per batch: %.3f, %.3f, %.3f' %
                      (epoch + 1, (i + 1)*args.batch_size, len(train_loader.dataset), losses_per_batch[-1], accuracy_per_batch[-1], f1_per_batch[-1]))
                total_loss = 0.0; total_acc = 0.0; total_f1 = 0.0
        
        scheduler.step()

        #test on val data
        results = evaluate_results(net, val_loader, pad_id, cuda)
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        f1_per_epoch.append(sum(f1_per_batch)/len(f1_per_batch))
        val_f1_per_epoch.append(results['f1'])
        print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        print("Train f1 at Epoch %d: %.7f" % (epoch + 1, f1_per_epoch[-1]))
        print("Val f1 at Epoch %d: %.7f" % (epoch + 1, val_f1_per_epoch[-1]))
        
        if val_f1_per_epoch[-1] > best_f1:
            best_f1 = val_f1_per_epoch[-1]
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_f1': val_f1_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict()
                }, os.path.join(args.model_save_path, "task_val_model_best_%d.pth.tar" % args.model_no))
        
        if (epoch % 1) == 0:
            save_as_pickle("task_train_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch, args.model_save_path)
            save_as_pickle("task_train_f1_per_epoch_%d.pkl" % args.model_no, f1_per_epoch, args.model_save_path)
            save_as_pickle("task_val_f1_per_epoch_%d.pkl" % args.model_no, val_f1_per_epoch, args.model_save_path)
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_f1': f1_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict()
                }, os.path.join(args.model_save_path, "task_train_checkpoint_%d.pth.tar" % args.model_no))
    
    logger.info("Finished Training!")
    return val_f1_per_epoch