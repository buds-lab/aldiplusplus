import warnings
import os
import sys
import logging
import yaml
import wandb
import torch
import pandas as pd
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score
from datetime import timedelta
from collections import Counter
from matplotlib import pyplot as plt

from utils import load_variable, save_variable, get_data_chunks
from vae import VAE
from data_import_ashrae import DataImportAshrae


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# load config file from CLI
with open(str(sys.argv[1]), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# extract parameters from config file
name = config["name"]
seed = config["seed"]

meter_type = config["meter_type"]
data_folds = config["data_folds"]

num_input = config["num_input"]
latent_dim = config["latent_dim"]
batch_size = config["batch_size"]
hidden_size = config["hidden_size"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]

# global variables
k_range = range(2, 11)

# starting wandb
wandb.init(project=name, entity="matiasqr")
config = wandb.config
config.data_folds = data_folds
config.latent_dim = latent_dim
config.batch_size = batch_size
config.hidden_size = hidden_size
config.learning_rate = learning_rate
config.epochs = epochs

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
print("Loading data ...")
df_metadata = DataImportAshrae().get_meta_data()
site_list = df_metadata["site_id"].unique()

site_list = [1] # DEBUG

for site in site_list:
    print(f"Site {site} ...")
    df_all = DataImportAshrae().get_daily_profiles(meter_type, [site])

    # prepare site data
    train_folds, test_folds, scaler = get_data_chunks(df_all, folds=data_folds)
    df_exportable = {}

    for fold in range(0, data_folds):
        train_loader = torch.utils.data.DataLoader(
            train_folds[fold].to_numpy(),
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_folds[fold].to_numpy(),
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed,
            drop_last=True,
        )

        # model
        model = VAE(num_input, latent_dim, hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        wandb.watch(model)

        # training
        print("Training model ...")
        codes = dict(mu=list(), log_sigma2=list())
        for epoch in range(0, epochs + 1):
            # Training
            if epoch > 0:
                model.train()
                train_loss = 0
                for _, x in enumerate(train_loader):
                    x = x[:,0:-1].to(device)
                    # forward
                    x_hat, mu, logvar = model(x.float())
                    loss = model.loss_function(x_hat.float(), x.float(), mu, logvar)
                    train_loss += loss.item()
                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # log
                wandb.log({f"train_loss_site{site}_fold{fold}": train_loss / len(train_loader.dataset)})

            # Testing
            means, logvars, labels = list(), list(), list()
            with torch.no_grad():
                model.eval()
                test_loss = 0
                for _, x in enumerate(test_loader):
                    x = x[:,0:-1].to(device)
                    # forward
                    x_hat, mu, logvar = model(x.float())
                    test_loss += model.loss_function(
                        x_hat.float(), x.float(), mu, logvar
                    ).item()
                    # log
                    means.append(mu.detach())
                    logvars.append(logvar.detach())
            # log
            codes["mu"].append(torch.cat(means))
            codes["log_sigma2"].append(torch.cat(logvars))
            test_loss /= len(test_loader.dataset)
            wandb.log({f"test_loss_site{site}_fold{fold}": test_loss})
            # end of training loop

        # latent space clustering with different k
        print("Latent space clustering ...")
        mu, _ = model.encode(torch.from_numpy(test_folds[fold].iloc[:,0:-1].to_numpy()).float())
        ssi_list = []
        for k in k_range:
            clust_algo = KMeans(n_clusters=k, random_state=seed).fit(mu.detach())
            labels = clust_algo.predict(mu.detach())
            ssi = silhouette_score(mu.detach(), labels)
            ssi_list.append(ssi)
            wandb.log({f"ssi_site{site}_fold{fold}": ssi, 'k': k})

        # latent space clustering with unique k
        k = 2 # NOTE: replace accordingly
        clust_algo = KMeans(n_clusters=k, random_state=seed).fit(mu.detach())
        labels = clust_algo.predict(mu.detach())

        # find the cluster with the least number of members
        print("Finding anomalies ...")
        dict_label_members = Counter(labels)
        min_cluster = min(dict_label_members, key=dict_label_members.get)

        # get indices of days that are members of this cluster
        test_data_label = test_folds[fold].copy()
        test_data_label['label'] = labels # append cluster label to data
        test_data_label = test_data_label[test_data_label['label'] == min_cluster]
        df_pred_labels = test_data_label.copy().reset_index(drop=False).rename(columns={"index":"timestamp"})[["timestamp","building_id"]]
        df_pred_labels["is_discord"] = 1
        df_pred_labels["meter"] = meter_type

        # use proper format following original train data
        df_left_keys = DataImportAshrae().get_train_data()
        df_left_keys["timestamp"] = df_left_keys["timestamp"].astype("datetime64[ns]")
        df_exportable[fold] = pd.merge(df_left_keys, df_pred_labels, how="left", on=["building_id", "meter", "timestamp"])
        df_exportable[fold]["is_discord"] = df_exportable[fold]["is_discord"].fillna(0) # NaNs are padded with 0
        df_exportable[fold]["is_discord"] = df_exportable[fold]["is_discord"].astype("int8")

        print(f"Transforming {(df_exportable[fold][df_exportable[fold]['is_discord'] == 1]).shape[0]} daily discords to hourly ...")
        # fill out remaining hours of a discord day as discords
        for idx, row in df_exportable[fold][df_exportable[fold]["is_discord"] == 1].iterrows():
            for h in range(1, 24):
                new_time = row["timestamp"] + timedelta(hours=h)
                base_idx = df_exportable[fold].index[(df_exportable[fold]["timestamp"] == new_time) & (df_exportable[fold]["meter"] == row["meter"]) & (df_exportable[fold]["building_id"] == row["building_id"])]
                df_exportable[fold].loc[base_idx, "is_discord"] = 1

        # end of data_folds loop
    
    print("Merging all folds discords ...")
    df_final_exportable = df_exportable[0].copy()
    for fold in range(0, data_folds):
        df_final_exportable["is_discord"] = df_final_exportable["is_discord"] | df_exportable[fold]["is_discord"]

    # export here now the 'is_discord'
    df_final_exportable["is_discord"].to_csv(f'data/pred_discord/discords_{name}.csv', index=False) 

    print(df_final_exportable[df_final_exportable["is_discord"] == 1])
    # TODO: only export site for each buildin_site
    
    # end of site loop