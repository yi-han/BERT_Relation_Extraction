from .train_funcs import evaluate_results, load_state
import logging
import torch

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def test(args, net, test_loader, tokenizer):
    logger.info("Restoring trained model...")
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    load_state(net, None, None, args, load_best=True)
    logger.info("Start testing...")
    results = evaluate_results(net, test_loader, tokenizer.pad_token_id, cuda)
    logger.info("Finished testing.")
    return results