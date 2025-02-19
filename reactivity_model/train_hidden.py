import torch

from data import split_data_training, normalize_data_hidden, PredDataset_hidden, get_metrics, write_predictions, scaling_back
import pandas as pd
from torch.utils.data import DataLoader
from ffn_model import ForwardFFN

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from args import get_args
from utils import setup_logger, ConsoleLogger
import logging

from pathlib import Path
import yaml
import pickle
import numpy as np

if __name__ == '__main__':

    # initialize
    args = get_args()
    save_dir = args.save_dir
    rxn_id_column = args.rxn_id_column

    # logger
    setup_logger(save_dir, log_name="ffn_train.log", debug=args.debug)
    pl.utilities.seed.seed_everything(args.random_state)

    # Dump args
    yaml_args = yaml.dump(args, indent=2, default_flow_style=False)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    logging.info(f"Args:\n{yaml_args}")

    predicted_activation_energies_ind = []
    # make a directory to store model files
    store_dir = Path(save_dir) / f"upper-bound"
    store_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.ensemble_size):
        logging.info(f"Training of model {i} started...")

        # load data set
        if args.data_path is not None:
            df = pd.read_pickle(args.data_path)# == train_valid_set_path
            # split data
            train, valid, _ = split_data_training(df, rxn_id_column, [0.9, 0.1, 0.0], args.random_state, i)
            test = pd.read_pickle(args.test_set_path)
        else:
            # NOTE Val on testing-set 
            train = pd.read_pickle(args.train_valid_set_path)
            valid = pd.read_pickle(args.test_set_path)
            test = valid

        

        # column-wise normalize data (each scaler for each column): removed currently
        train_scaled, train_scalers = normalize_data_hidden(train, rxn_id_column, args.target_column)
        valid_scaled, _ = normalize_data_hidden(valid, rxn_id_column, args.target_column, scalers=train_scalers)
        test_scaled, _ = normalize_data_hidden(test, rxn_id_column, args.target_column, scalers=train_scalers)

        # process data NOTE scaled!!!!!!
        train_dataset = PredDataset_hidden(
            train_scaled,
            args.features,
            args.target_column,
        )
        valid_dataset = PredDataset_hidden(
            valid_scaled,
            args.features,
            args.target_column,
        )
        test_dataset = PredDataset_hidden(
            test_scaled,
            args.features,
            args.target_column,
        )

        if args.data_path is not None:
            dataset_sizes = (len(train_dataset), len(valid_dataset), len(test_dataset))
            logging.info(f"Train, val, test: {dataset_sizes}")
        else:
            dataset_sizes = (len(train_dataset), len(test_dataset))
            logging.info(f"Train, test: {dataset_sizes}. Using test-set to early-stop(test=val)")

        # Define dataloaders
        train_loader = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size=args.batch_size)
        val_loader = DataLoader(valid_dataset,
                                shuffle=False,
                                batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset,
                                shuffle=False,
                                batch_size=args.batch_size)

        # Define model
        test_batch = next(iter(train_loader))

        input_dim = train_loader.dataset.descriptors.shape[-1]

        if args.transfer_learning:
            trained_checkpoint = Path(args.trained_dir) / f"best_model_{i}.ckpt"
            model = ForwardFFN.load_from_checkpoint(trained_checkpoint)
            # for name, param in model.named_parameters():
            #    if param.requires_grad and 'all_layers.1.0.' in name:
            #        continue
            #    param.requires_grad = False
        else:
            model = ForwardFFN(
                hidden_size=args.hidden_size,
                layers=args.layers,
                dropout=args.dropout,
                input_dim=input_dim,
                output_dim=1,
                learning_rate=args.learning_rate,
                lr_ratio=args.lr_ratio
            )

        # Create trainer
        tb_logger = pl_loggers.TensorBoardLogger(store_dir, name="", version="")
        console_logger = ConsoleLogger()

        tb_path = tb_logger.log_dir
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",#TODO
            dirpath=tb_path,
            filename=f"best_model_{i}",
            save_weights_only=False,
        )
        earlystop_callback = EarlyStopping(monitor="val_loss", patience=10)
        callbacks = [earlystop_callback, checkpoint_callback]

        trainer = pl.Trainer(logger=tb_logger,
                             accelerator="gpu" if args.gpu else "cpu",
                             gpus=1 if args.gpu else 0,
                             callbacks=callbacks,
                             gradient_clip_val=5,
                             max_epochs=args.max_epochs,
                             gradient_clip_algorithm="value")

        trainer.fit(model, train_loader, val_loader)
        checkpoint_callback = trainer.checkpoint_callback
        best_checkpoint = checkpoint_callback.best_model_path
        best_checkpoint_score = checkpoint_callback.best_model_score.item()

        # Load from checkpoint
        model = ForwardFFN.load_from_checkpoint(best_checkpoint)

        model.eval()
        test_out = trainer.test(model=model, dataloaders=test_loader)
        test_pred = trainer.predict(model=model, dataloaders=test_loader)

        out_yaml = {"args": args, "test_metrics": test_out[0]}
        out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)
        with open(Path(save_dir) / "test_results.yaml", "w") as fp:
            fp.write(out_str)

        pred_values_TS = scaling_back(test_pred, train_scalers[args.target_column])

        predicted_activation_energies_ind.append(pred_values_TS)

    # determine ensemble predictions
    predicted_activation_energies = np.sum(predicted_activation_energies_ind, axis=0) / len(predicted_activation_energies_ind)

    # write predictions for fold i to csv file
    write_predictions(
        test.rxn_id,
        predicted_activation_energies,
        args.rxn_id_column,
        Path(save_dir) / f"test_predicted_{i}.csv",)

    # compute and store metrics
    mae_act, rmse_act, r2_act = get_metrics(test[args.target_column], predicted_activation_energies)
    
    # report final results at the end of the run
    logging.info(
        f"RMSE for upper-bound prediction - {args.target_column}: "
        f"{rmse_act}"
        f"\nMAE for upper-bound prediction - {args.target_column}: "
        f"{mae_act}"
        f"\nR2 for upper-bound prediction - {args.target_column}: "
        f"{r2_act}"
    )