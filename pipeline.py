from src.preprocessing_funcs import Pad_Sequence, load_pretrained_model, load_dataset
from src.cross_val import prepare_train_val_data
from src.train import train_and_fit
from src.test import test
from src.infer import infer_from_trained
from torch.utils.data import DataLoader
import logging
from argparse import ArgumentParser
import copy
import os

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='oscer', help='oscer')
    parser.add_argument("--train_data", type=str, default='./data/oscer/train.tsv', \
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='./data/oscer/test.tsv', \
                        help="test data .txt file path")
    parser.add_argument("--num_classes", type=int, default=2, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--num_epochs", type=int, default=5, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00002, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT''')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
                                                                                                'bert-large-uncased',\
                                                                                    For ALBERT: 'albert-base-v2',\
                                                                                                'albert-large-v2'")
    parser.add_argument("--model_save_path", type=str, default='./trained_model/', help="folder for saving the model")
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--test", type=int, default=1, help="0: Don't test, 1: test")
    parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")
    
    args = parser.parse_args()

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    #load pretrained BERT/ALBERT model and the corresponding tokenizer
    net, tokenizer = load_pretrained_model(args)

    #load train and test data
    train_set, test_set = load_dataset(args, tokenizer)
    PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id,\
                    label_pad_value=tokenizer.pad_token_id,\
                    label2_pad_value=-1)

    if args.train == 1:
        #k-fold cross validation
        k = 10
        val_f1 = []
        for i in range(k):
            train_loader, val_loader = prepare_train_val_data(copy.deepcopy(train_set), PS, args.batch_size, i, k)
            val_f1_per_epoch = train_and_fit(args, net, train_loader, val_loader, tokenizer)
            val_f1.append(val_f1_per_epoch)
            net, _ = load_pretrained_model(args)
    if args.test == 1:
        #test
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, \
                                num_workers=0, collate_fn=PS, pin_memory=False)
        results = test(args, net, test_loader, tokenizer)

    if args.infer == 1:
        inferer = infer_from_trained(args, net, tokenizer)
        
        while True:
            sent = input("Type input sentence ('quit' or 'exit' to terminate):\n")
            if sent.lower() in ['quit', 'exit']:
                break
            inferer.infer_sentence(sent)