import torch

import ffn_model
from data import split_data_cross_val, normalize_data_hidden, scaling_back, PredDataset_hidden
from data import get_metrics, write_predictions, delta_target
import pandas as pd
import numpy as np
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

if __name__ == '__main__':

    # initialize
    args = get_args(cross_val=True)
    # args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
    # args.features = ['mol1_hidden', 'rad1_hidden', 'rad_atom1_hidden']
    save_dir = args.save_dir
    trained_dir = args.trained_dir
    splits_dir = Path(save_dir) / "splits"
    rxn_id_column = args.rxn_id_column

    # load data set
    if args.data_path is not None:
        df = pd.read_pickle(args.data_path)
    else:
        # df = pd.read_csv(args.train_valid_set_path, index_col=0)
        df = pd.read_pickle(args.train_valid_set_path)

    df = df.sample(frac=1, random_state=args.random_state)

    # logger
    setup_logger(save_dir, log_name="ffn_train.log", debug=args.debug)
    pl.utilities.seed.seed_everything(args.random_state)

    # Dump args
    yaml_args = yaml.dump(args, indent=2, default_flow_style=False)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    logging.info(f"Args:\n{yaml_args}")

    # split df into k_fold groups
    k_fold_arange = np.linspace(0, len(df), args.k_fold + 1).astype(int)

    # create lists to store metrics for each fold
    rmse_activation_energy_list = []
    mae_activation_energy_list = []
    r2_activation_energy_list = []

    if args.delta_ML:
        rmse_activation_energy_delta_list = []
        mae_activation_energy_delta_list = []
        r2_activation_energy_delta_list = []

    for i in range(args.k_fold):

        logging.info(f"Training the {i}th iteration...")

        # make a directory to store model files
        fold_dir = Path(save_dir) / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # initialize list to store predictions for each model in the ensemble
        predicted_activation_energies_ind = []
        predicted_reaction_energies_ind = []

        # within a fold, loop over the ensemble size (default -> 1)
        for j in range(args.ensemble_size):
            if args.ensemble_size > 1:
                logging.info(f"Training of model {j} started...")
            train, valid, test = split_data_cross_val(
                df,
                k_fold_arange,
                i,
                j,
                args.rxn_id_column,
                args.data_path,
                args.csv_file,
                args.train_valid_set_path,
                args.sample,
                args.k_fold,
                args.random_state,
                args.test_set_path,
            )

            if args.delta_ML:
                train, valid, test = delta_target(train, valid, test)

            # only store splits when a single model is used due to
            # ballooning storage footprint
            if args.ensemble_size == 1:
                current_split_dir = Path(splits_dir) / f"fold_{i}"
                current_split_dir.mkdir(parents=True, exist_ok=True)
                train.to_csv(Path(current_split_dir) / 'train.csv')
                valid.to_csv(Path(current_split_dir) / 'valid.csv')
                test.to_csv(Path(current_split_dir) / 'test.csv')

            logging.info(
                f" Size train set: {len(train)} - size validation set: {len(valid)} - size test set: {len(test)}"
            )

            # normalize data TODO is suitable???
            # Only target is scaled for now
            train_scaled, train_scalers = normalize_data_hidden(train, args.rxn_id_column, args.target_column)
            valid_scaled, _ = normalize_data_hidden(valid, args.rxn_id_column, args.target_column, scalers=train_scalers)
            test_scaled, _ = normalize_data_hidden(test, args.rxn_id_column, args.target_column, scalers=train_scalers)
            
            scalers_dir_path = Path(save_dir) / f"fold_{i}/scalers_{j}"
            scalers_dir_path.mkdir(parents=True, exist_ok=True)
            pickle.dump(
            train_scalers,
            open(
                Path(scalers_dir_path) / f"scaler_{j}.pickle",
                "wb",
            ),
        )

            # process data
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

            input_dim = train_dataset.getdimension()

            if args.transfer_learning:
                trained_checkpoint = Path(trained_dir) / f"best_model_{j}.ckpt"
                model = ForwardFFN.load_from_checkpoint(trained_checkpoint)
            else:
                model = ForwardFFN(
                    hidden_size=args.hidden_size,
                    layers=args.layers,
                    dropout=args.dropout,
                    learning_rate=args.learning_rate,
                    input_dim=input_dim,
                    output_dim=1,
                    lr_ratio=args.lr_ratio
                )

            # Create trainer
            tb_logger = pl_loggers.TensorBoardLogger(fold_dir, name="", version="")
            console_logger = ConsoleLogger()

            tb_path = tb_logger.log_dir
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=tb_path,
                filename=f"best_model_{j}",
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
                                 gradient_clip_algorithm="value",
                                 )

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
            # NOTE 除了还原归一化，还做了数据整合

            predicted_activation_energies_ind.append(pred_values_TS)

        # determine ensemble predictions

        predicted_activation_energies = \
            np.sum(predicted_activation_energies_ind, axis=0) / len(predicted_activation_energies_ind)

        # write predictions for fold i to csv file
        write_predictions(
            test.rxn_id,
            predicted_activation_energies,
            args.rxn_id_column,
            Path(save_dir) / f"test_predicted_{i}.csv",)

        # compute and store metrics
        mae_act, rmse_act, r2_act = get_metrics(test[args.target_column], predicted_activation_energies)

        rmse_activation_energy_list.append(rmse_act)
        mae_activation_energy_list.append(mae_act)
        r2_activation_energy_list.append(r2_act)

        logging.info(
            f"success rate for iter {i} - activation energy: {rmse_act}, {mae_act}, {r2_act}"
        )

        if args.delta_ML:
            test['ddG_predict'] = predicted_activation_energies
            test['dG_TS_tunn_delta_predict'] = test['ddG_predict'] + test['DG_TS_tunn_linear']
            # compute and store metrics
            mae_act, rmse_act, r2_act = get_metrics(test['DG_TS_tunn'], test['dG_TS_tunn_delta_predict'])

            rmse_activation_energy_delta_list.append(rmse_act)
            mae_activation_energy_delta_list.append(mae_act)
            r2_activation_energy_delta_list.append(r2_act)

    
    # report final results at the end of the run
    logging.info(
        f"RMSE for {args.k_fold}-fold cross-validation - {args.target_column}: "
        f"{np.mean(np.array(rmse_activation_energy_list))}"
        f"\nMAE for {args.k_fold}-fold cross-validation - {args.target_column}: "
        f"{np.mean(np.array(mae_activation_energy_list))}"
        f"\nR2 for {args.k_fold}-fold cross-validation - {args.target_column}: "
        f"{np.mean(np.array(r2_activation_energy_list))}"
    )
    
    # cv = {f"RMSE for {args.k_fold}-fold cross-validation - {args.target_column}: " : np.mean(np.array(rmse_activation_energy_list)),
    #     f"\nMAE for {args.k_fold}-fold cross-validation - {args.target_column}: " : np.mean(np.array(mae_activation_energy_list)),
    #     f"\nR2 for {args.k_fold}-fold cross-validation - {args.target_column}: " : np.mean(np.array(r2_activation_energy_list))}
    # out_yaml = {"args": args, f"{args.k_fold}-fold cross-validation results": cv}
    # out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)

    # with open(Path(save_dir) / "cv_results_hyper_opt.yaml", "a") as fp:
    #     fp.write(out_str)

    if args.delta_ML:
        # report final results at the end of the run
        logging.info(
            f"RMSE for {args.k_fold}-fold cross-validation - delta-ML - activation energy: "
            f"{np.mean(np.array(rmse_activation_energy_delta_list))}"
            f"\nMAE for {args.k_fold}-fold cross-validation - delta-ML - activation energy: "
            f"{np.mean(np.array(mae_activation_energy_delta_list))}"
            f"\nR2 for {args.k_fold}-fold cross-validation - delta-ML - activation energy: "
            f"{np.mean(np.array(r2_activation_energy_delta_list))}"
        )






