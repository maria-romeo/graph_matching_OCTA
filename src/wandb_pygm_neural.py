import wandb
from torch.utils.data import DataLoader
from datasets.graph_matching_dataset import GraphMatchingDataset
import os
import pygmtools as pygm
import torch
from tools.calculate_metrics import calculate_metrics
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import Subset
from tools.train_neural_models import train, validate
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, '..'))
from visualization_tools.vvg_loader import vvg_to_df


valid_feats1 = ['length', 'distance', 'curveness', 'volume', 'avgCrossSection', 'minRadiusAvg', 'avgRadiusAvg', 'roundnessAvg']
valid_feats2 = ['length', 'distance', 'volume', 'avgCrossSection', 'avgRadiusAvg']

## WANDB ##
# Define sweep config
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "best_acc"},
    "parameters": {
        "model": {"values": ['cie']}, # 'ipca_gm', 'pca_gm', 'ngm', 'cie'
        "dataset": {"values": ['soul_big']}, # 'soul'
        "valid_feats": {"values": [valid_feats1, valid_feats2]},#,,['avgCrossSection', 'avgRadiusAvg']]},#,['length'], ['distance'], ['curveness'], ['volume'], ['avgCrossSection'], ['minRadiusAvg'], ['avgRadiusAvg'], ['roundnessAvg']},
        "epochs": {"values": [20]},
        "lr": {"values": [10e-6, 10e-7]},
        "momentum": {"values": [0.99, 0,9]},
        "weight_decay": {"values": [1e-4, 1e-6]},
        "optimizer": {"values": ['adamW']},#,, 'sgd']},
        
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="GM_pygm_neural_networks")


i = 0
def main():
    global i 
    i +=1
    run = wandb.init()
    name = f"{i}_{wandb.config.model}_dataset{wandb.config.dataset}_lr{wandb.config.lr}"
    wandb.run.name = name
    api = wandb.Api()
    sweep = api.sweep("ge92lik/GM_pygm_neural_networks/" + sweep_id)

    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print GPU information at the beginning
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU.")

    if wandb.config.dataset == 'soul':
        dataset_name = 'soul_dataset'
    elif wandb.config.dataset == 'soul_big':
        dataset_name = 'soul_dataset_big'
    src_dir = f'../data/{dataset_name}/'
    transforms_dir = os.path.join(src_dir, 'transformed_images')

    # Initialize the dataset
    dataset = GraphMatchingDataset(src_dir=src_dir, 
                                valid_feats=wandb.config.valid_feats, 
                                transforms_dir=transforms_dir)
    
    # Group images by subject
    subject_to_images = defaultdict(list)
    for index in dataset.indexes:
        subject = index.split('_')[0]  # Extract the subject number
        subject_to_images[subject].append(index)

    # Convert grouped data to a list of subject groups for splitting
    subject_groups = list(subject_to_images.values())

    train_subjects, tmp_subjects = train_test_split(subject_groups, test_size=0.40, random_state=42)
    val_subjects, test_subjects = train_test_split(tmp_subjects, test_size=0.50, random_state=42)

    # Flatten lists of images per split
    train_idxs = [img for subject in train_subjects for img in subject]
    val_idxs = [img for subject in val_subjects for img in subject]
    test_idxs = [img for subject in test_subjects for img in subject]

    # Convert the indexes back to actual dataset indices
    train_set = Subset(dataset, [dataset.indexes.index(idx) for idx in train_idxs])
    val_set = Subset(dataset, [dataset.indexes.index(idx) for idx in val_idxs])
    test_set = Subset(dataset, [dataset.indexes.index(idx) for idx in test_idxs])

    # Create DataLoaders for each set
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


    pygm.set_backend('pytorch')


    # Create the network using pygmtools' get_network
    if wandb.config.model == 'ipca_gm':
        net = pygm.utils.get_network(pygm.ipca_gm, pretrain='voc').to(device)
    elif wandb.config.model == 'pca_gm':
        net = pygm.utils.get_network(pygm.pca_gm, pretrain='voc').to(device)
    elif wandb.config.model == 'ngm':
        net = pygm.utils.get_network(pygm.ngm, pretrain='voc').to(device)
    elif wandb.config.model == 'cie':
        net = pygm.utils.get_network(pygm.cie, pretrain='voc').to(device)
    
    for param in net.parameters():
        param.requires_grad = True


    # Set up an optimizer (SGD is used here as in the example, but you could use Adam or others)
    optimizer_dict = {"adamW": torch.optim.AdamW(net.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay),
                        "sgd": torch.optim.SGD(net.parameters(), lr=wandb.config.lr, momentum=wandb.config.momentum, weight_decay=wandb.config.weight_decay)}

    optimizer = optimizer_dict[wandb.config.optimizer]

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 5  # Stop after 5 epochs without improvement

    # Store best accuracy
    best_acc = 0

    for epoch in range(wandb.config.epochs):
        
        # train
        net, avg_loss, acc, prec, recall, f1 = train(train_loader, net.to(device), wandb.config.model, optimizer, device)

        mean_acc = torch.tensor(acc).mean()
        mean_prec = torch.tensor(prec).mean()
        mean_recall = torch.tensor(recall).mean()
        mean_f1 = torch.tensor(f1).mean()
        std_acc = torch.tensor(acc).std()
        std_prec = torch.tensor(prec).std()
        std_recall = torch.tensor(recall).std()
        std_f1 = torch.tensor(f1).std()

        # Print average loss for this epoch
        print(f'Epoch {epoch+1}/{wandb.config.epochs}, Loss: {avg_loss}, Acc: {mean_acc}, Prec: {mean_prec}, Recall: {mean_recall}, F1: {mean_f1}')

        # validate
        net, avg_val_loss, val_acc, val_prec, val_recall, val_f1 = validate(val_loader, net.to(device), wandb.config.model, device)

        mean_val_acc = torch.tensor(val_acc).mean()
        mean_val_prec = torch.tensor(val_prec).mean()
        mean_val_recall = torch.tensor(val_recall).mean()
        mean_val_f1 = torch.tensor(val_f1).mean()
        std_val_acc = torch.tensor(val_acc).std()   
        std_val_prec = torch.tensor(val_prec).std()
        std_val_recall = torch.tensor(val_recall).std()
        std_val_f1 = torch.tensor(val_f1).std()

        if mean_val_acc > best_acc:
            best_acc = mean_val_acc

        wandb.log({"loss": avg_loss,
                   " best val acc": best_acc,
                    "train_acc": mean_acc, "train_acc_std": std_acc,
                    "train_prec": mean_prec, "train_prec_std": std_prec,
                    "train_recall": mean_recall, "train_recall_std": std_recall,
                    "train_f1": mean_f1, "train_f1_std": std_f1,
                    "val_loss": avg_val_loss,
                    "val_acc": mean_val_acc, "val_acc_std": std_val_acc,
                    "val_prec": mean_val_prec, "val_prec_std": std_val_prec,
                    "val_recall": mean_val_recall, "val_recall_std": std_val_recall,
                    "val_f1": mean_val_f1, "val_f1_std": std_val_f1
        })

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0  # Reset the counter if validation loss improves
        else:
            epochs_without_improvement += 1  # Increment the counter if no improvement
        
        # Stop training if no improvement in last `patience` epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

wandb.agent(sweep_id, function=main, count=10000)
