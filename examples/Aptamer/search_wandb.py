# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

import sys
from main import main
from args import ArgsInit
import random
import logging


# 1: Define objective/training function
def objective(config):
    """Function with unknown internals we wish to maximize.
    """
    args = ArgsInit().save_exp()
    args.num_layers = config.num_layers
    args.hidden_channels = config.hidden_channels
    args.gcn_aggr = config.gcn_aggr
    args.batch_size = config.hidden_channels
    
    args.epochs = 50
    args.use_gpu = True
    args.add_virtual_node= False

    return main(args)["highest_valid_pear"]

def search():
    wandb.init(project="gcn-ngs-capping-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})
    

# 2: Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "num_layers": {"min": 3, "max": 12},
        "hidden_channels": {"values": [64, 128, 256]},
        "gcn_aggr": {"values": ['max', 'mean', 'power', 'add', 'softmax', 'softmax_sg']},
        "batch_size": {"values":[64]},
        "lr": {"values": [0.01, 0.005, 0.001]}
    },
}


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="gcn-ngs-capping-sweep")
wandb.agent(sweep_id, function=search, count=100)