"""Trains a model on a dataset."""

from utilities.parsing import parse_train_args
from run.run_training import run_training
from utilities.utils_gen import create_logger, load_args
import pickle
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", default=1200, type=int, help='1200 by original authors')
    parser.add_argument("--batch_size", default=50, type=int, help='50 by original authors')
    parser.add_argument("--epochs", default=100, type=int, help='100 by original authors. NOTE: The training does not reach the early-stop point within 100 epochs.')
    emilien_args = parser.parse_args()
    
    default_train_args = load_args('qmdesc_wrap/model_original.pt')
    
    default_train_args.hidden_size = emilien_args.hidden_size
    default_train_args.gpu = 0
    default_train_args.cuda = True
    default_train_args.batch_size = emilien_args.batch_size
    default_train_args.epochs = emilien_args.epochs
    default_train_args.save_dir = f"output_h{emilien_args.hidden_size}_b{emilien_args.batch_size}_e{emilien_args.epochs}"
    
    args = default_train_args # TODO Using mostly the default args given by authors.
    
    
    # args = parse_train_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    avg_test_score, test_scores, test_preds, test_smiles = run_training(args, logger)
    with open(os.path.join(args.save_dir, 'test_preds.pickle'), 'wb') as preds:
        pickle.dump(test_preds, preds)

    with open(os.path.join(args.save_dir, 'test_smiles.pickle'), 'wb') as smiles:
        pickle.dump(test_smiles, smiles)
