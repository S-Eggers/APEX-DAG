import optuna
import os
import yaml
from pathlib import Path
from ApexDAG.experiments.pretrain import pretrain_gat
from ApexDAG.util.logging import setup_logging, setup_wandb


def objective(trial):
    
    logger = setup_logging("hyperparam_optim", True)
    
    config = {
        "checkpoint_path": "/home/eggers/data/apexdag_results/jetbrains_dfg_100k_new/execution_graphs",
        "encoded_checkpoint_path": "data/raw/",
        "node_classes": 8,
        "edge_classes": 6,
        "num_epochs": 50,
        "train_split": 0.8,
        "patience": 10,
        'seed': 42,
        # hyperparameters
        "residual": trial.suggest_categorical("residual", [True, False]),
        "number_gat_blocks": trial.suggest_int("number_gat_blocks", 2, 4),
        "batch_size": trial.suggest_categorical("batch_size", [32]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.001, 0.005]),
        "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0.2, 0.4, 0.6]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.00001, 0.0001]),
        "dim_embed": 300,

        "min_nodes": 3,
        "min_edges": 2,

        "load_encoded_old_if_exist": True,
        "device": "cuda"
    }

    hash_value = hash(str(config))
    os.makedirs("hyperparam_optim", exist_ok = True)

    setup_wandb(project_name="APEX-DAG-hyperparam-optimization", name = hash_value)
    print(config)

    with open(f"hyperparam_optim/config_{hash_value}.yaml", "w") as f:
        yaml.dump(config, f)

    args = {'config_path': f"hyperparam_optim/config_{hash_value}.yaml"}
    best_val_loss = pretrain_gat(args, logger)
    
    with open("hyperparam_optim/trial_results.txt", "a") as f:
        f.write(f"Trial {hash_value} - Best Validation Loss: {best_val_loss}\n")
    
    return best_val_loss

def save_best_config(best_trial):
    """Save the best configuration and its losses."""
    best_config = best_trial.params
    best_losses = best_trial.user_attrs["best_val_loss"]

    config_path = Path("best_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(best_config, f)

    loss_file = Path("best_losses.txt")
    with open(loss_file, "w") as f:
        f.write(f"Best Validation Loss: {best_losses}")
        
if __name__ == "__main__":
        
    study = optuna.create_study(direction="minimize")  # Minimize validation loss
    study.optimize(objective, n_trials=1000)

    best_trial = study.best_trial
    save_best_config(best_trial)
