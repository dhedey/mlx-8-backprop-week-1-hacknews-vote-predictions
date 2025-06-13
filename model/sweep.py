#!/usr/bin/env python3
"""
Weights & Biases Hyperparameter Sweep Script

This script programmatically creates and runs wandb sweeps for the HackerNews score prediction model.
It provides more control over the sweep process compared to the CLI-based approach.
"""

import wandb
import os

from model.model import TrainingHyperparameters
from train import train_model
from model import ModelHyperparameters

# Sweep configuration - equivalent to wandb_sweep.yaml but in Python
SWEEP_CONFIG = {
    'method': 'random',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [1024]
        },
        'freeze_embeddings': {
            'values': [True, False]
        },
        'include_batch_norms': {
            'values': [True, False]
        },
        'learning_rate': {
            'min': 0.001,
            'max': 0.03,
            'distribution': 'log_uniform_values'
        },
        'dropout': {
            'min': 0.1,
            'max': 0.5,
            'distribution': 'uniform'
        },
        'hidden_dim_1': {
            'values': [64, 128, 256, 512]
        },
        'hidden_dim_2': {
            'values': [64, 128, 256, 512]
        },
        'hidden_dim_3': {
            'values': [32, 64, 128, 256, 512]
        },
        'epochs': {
            'values': [2, 4]
        }
    }
}

def train_sweep_run():
    """
    Single training run for wandb sweep.
    This function is called by the sweep agent for each hyperparameter combination.
    """
    # Initialize wandb run
    wandb.init()
    
    try:
        config = wandb.config
        
        print(f"\n🚀 Starting sweep run")

        results = train_model(
            model_parameters=ModelHyperparameters(
                hidden_dimensions=[config.hidden_dim_1, config.hidden_dim_2, config.hidden_dim_3],
                include_batch_norms=config.include_batch_norms,
            ),
            training_parameters=TrainingHyperparameters(
                batch_size=config.batch_size,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                freeze_embeddings=config.freeze_embeddings,
                dropout=config.dropout,
            ),
        )
        
        # Log final metrics (wandb.log is also called within train_model)
        wandb.log({
            "final_train_loss": results['final_train_loss'],
            "final_test_loss": results['final_test_loss'],
            "best_test_loss": results['best_test_loss'],
            "epochs_completed": results['epochs_completed']
        })
        
        print(f"✅ Sweep run completed! Test loss: {results['final_test_loss']:.4f}")
        
    except Exception as e:
        print(f"❌ Sweep run failed: {e}")
        # Log the failure
        wandb.log({"status": "failed", "error": str(e)})
        raise
    
    finally:
        # Ensure wandb run is properly finished
        wandb.finish()


def create_and_run_sweep(config, project_name, count=10):
    """
    Create and run a wandb sweep programmatically.
    
    Args:
        config: Sweep configuration dictionary (defaults to SWEEP_CONFIG)
        project_name: W&B project name
        count: Number of runs to execute in the sweep
    """
    print(f"🔧 Creating sweep with {config['method']} optimization...")
    print(f"📊 Target metric: {config['metric']['name']} ({config['metric']['goal']})")
    
    # Create the sweep
    sweep_id = wandb.sweep(config, project=project_name)
    print(f"✅ Sweep created with ID: {sweep_id}")
    print(f"🌐 View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    # Run the sweep
    print(f"🏃 Starting sweep agent with {count} runs...")
    wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)
    
    print(f"🎉 Sweep completed!")
    return sweep_id


def run_existing_sweep(sweep_id, project_name, count=10):
    """
    Run an existing sweep by ID.
    
    Args:
        sweep_id: The ID of an existing sweep
        count: Number of additional runs to execute
    """
    print(f"🔄 Joining existing sweep: {sweep_id} against {project_name}")
    wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)


def main():
    """
    Main function with different sweep options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter sweeps for HackerNews model')
    parser.add_argument('--project', default='hackernews-score-prediction',
                        help='W&B project name (default: hackernews-score-prediction)')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of sweep runs (default: 20)')
    parser.add_argument('--sweep-id', type=str,
                        help='Join existing sweep by ID instead of creating new one')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show the configuration without running')
    
    args = parser.parse_args()
    
    # Select configuration
    config = SWEEP_CONFIG
    print("📋 Using default Bayesian optimization configuration")
    
    if args.dry_run:
        print("\n🔍 Sweep configuration:")
        import json
        print(json.dumps(config, indent=2))
        return
    
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Run sweep
    if args.sweep_id:
        run_existing_sweep(args.sweep_id, args.project, args.count)
    else:
        sweep_id = create_and_run_sweep(config, args.project, args.count)
        print(f"\n💾 Save this sweep ID for future use: {sweep_id}")


if __name__ == '__main__':
    main()
