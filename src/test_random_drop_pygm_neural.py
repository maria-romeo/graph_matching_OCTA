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
from tools.train_neural_models import evaluate
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
        "model": {"values": ['pca_gm', 'ipca_gm', 'cie', 'ngm']}, # 'ipca_gm', 'pca_gm', 'ngm', 'cie'
        "dataset": {"values": ['random_25_drop', 'random_50_drop', 'random_75_drop', 'random_100_drop']}, # 'soul'
        "valid_feats": {"values": [valid_feats1]},#,valid_feats2,['avgCrossSection', 'avgRadiusAvg']]},#,['length'], ['distance'], ['curveness'], ['volume'], ['avgCrossSection'], ['minRadiusAvg'], ['avgRadiusAvg'], ['roundnessAvg']},
        "epochs": {"values": [20]},
        "lr": {"values": [10e-6]},
        "momentum": {"values": [0.99]},
        "weight_decay": {"values": [1e-4]}, #1e-6, 1e-4
        "optimizer": {"values": ['adamW']},#,, 'sgd']},
        
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="GM_test")


i = 0
def main():
    global i 
    i +=1
    run = wandb.init()
    name = f"{i}_{wandb.config.model}_dataset{wandb.config.dataset}_lr{wandb.config.lr}"
    wandb.run.name = name
    api = wandb.Api()
    sweep = api.sweep("ge92lik/GM_test/" + sweep_id)

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
    elif wandb.config.dataset == 'longitudinal':
        dataset_name = 'soul_longitudinal'
    elif wandb.config.dataset == 'longitudinal_post':
        dataset_name = 'soul_longitudinal_pre_to_post'
    elif 'random' in wandb.config.dataset:
            dataset_name = f'soul_random_drop_vessels/{wandb.config.dataset}'
   
    src_dir = f'../data/{dataset_name}/'
    transforms_dir = os.path.join(src_dir, 'transformed_images')

    # Initialize the dataset
    dataset = GraphMatchingDataset(src_dir=src_dir, 
                                valid_feats=wandb.config.valid_feats, 
                                transforms_dir=transforms_dir)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


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

    # Store best accuracy
    best_acc = 0

    # Store the evaluation metrics for every sample
    records = []
    # Create dirs to store results
    if not os.path.exists(f'GM_results_{wandb.config.dataset}'):
        os.makedirs(f'GM_results_{wandb.config.dataset}')
    if not os.path.exists(f'GM_results_{wandb.config.dataset}/pygm_neural/{wandb.config.model}_matrix_results'):
        os.makedirs(f'GM_results_{wandb.config.dataset}/pygm_neural/{wandb.config.model}_matrix_results')
    x_pred_dir = f'GM_results_{wandb.config.dataset}/pygm_neural/{wandb.config.model}_matrix_results'

    best_model_path = f'GM_results/pygm_neural/{wandb.config.model}_best_model.pth'
    # Load the best model for evaluation
    net.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    net.to(device)

    # Evaluate the model on the test set
    output_dir = f'GM_results_{wandb.config.dataset}'
    net, avg_test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate(dataloader, net, wandb.config.model, device, output_dir)

    mean_test_acc = torch.tensor(test_acc).mean()
    mean_test_prec = torch.tensor(test_prec).mean()
    mean_test_recall = torch.tensor(test_recall).mean()
    mean_test_f1 = torch.tensor(test_f1).mean()
    std_test_acc = torch.tensor(test_acc).std()
    std_test_prec = torch.tensor(test_prec).std()
    std_test_recall = torch.tensor(test_recall).std()
    std_test_f1 = torch.tensor(test_f1).std()

    # Log final test results to wandb
    wandb.log({
        "test_acc": mean_test_acc, "test_acc_std": std_test_acc,
        "test_prec": mean_test_prec, "test_prec_std": std_test_prec,
        "test_recall": mean_test_recall, "test_recall_std": std_test_recall,
        "test_f1": mean_test_f1, "test_f1_std": std_test_f1
    })
    
    # SAVE METRICS AND MODEL CONFIG
    # Save the model configuration
    # Write model and features to a text file
    with open(f'GM_results_{wandb.config.dataset}/pygm_neural/{wandb.config.model}_config_info.txt', 'a') as config_file:
        config_file.write(f"Model: {wandb.config.model}\n")
        config_file.write(f"Dataset: {dataset}\n")
        config_file.write(f"Features: {wandb.config.valid_feats}\n")
        config_file.write(f"Learing Rate: {wandb.config.lr}\n")
        config_file.write(f"Momentum: {wandb.config.momentum}\n")
        config_file.write(f"Weight Decay: {wandb.config.weight_decay}\n")
        config_file.write(f"Optimizer: {wandb.config.optimizer}\n")
        config_file.write("\n")

    # Save the evaluation metrics
    with open(f'GM_results_{wandb.config.dataset}/pygm_neural/{wandb.config.model}_avge_evaluation_metrics.txt', 'a') as metrics_file:
        metrics_file.write(f"Mean Test Accuracy: {mean_test_acc}, Mean Test Precision: {mean_test_prec}, Mean Test Recall: {mean_test_recall}, Mean Test F1 Score: {mean_test_f1}\n")
        metrics_file.write(f"Std Test Accuracy: {std_test_acc}, Std Test Precision: {std_test_prec}, Std Test Recall: {std_test_recall}, Std Test F1 Score: {std_test_f1}\n")
        metrics_file.write("\n")


wandb.agent(sweep_id, function=main, count=30)