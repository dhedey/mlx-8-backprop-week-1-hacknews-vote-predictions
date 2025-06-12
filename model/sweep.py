#!/usr/bin/env python3
"""
Weights & Biases Hyperparameter Sweep Script

This script programmatically creates and runs wandb sweeps for the HackerNews score prediction model.
It provides more control over the sweep process compared to the CLI-based approach.
"""

import wandb
import os
from train import train_model, ModelRunSettings


# Sweep configuration - equivalent to wandb_sweep.yaml but in Python
SWEEP_CONFIG = {
    'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [64, 128, 256]
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01,
            'distribution': 'log_uniform_values'
        },
        'dropout': {
            'min': 0.1,
            'max': 0.5,
            'distribution': 'uniform'
        },
        'hidden_dim_1': {
            'values': [128, 256, 512]
        },
        'hidden_dim_2': {
            'values': [256, 512, 1024]
        },
        'epochs': {
            'value': 5  # Fixed value for the sweep
        }
    }
}

# Alternative sweep configurations for different experiments
QUICK_SWEEP_CONFIG = {
    'method': 'grid',
    'metric': {
        'name': 'test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [128, 256]
        },
        'learning_rate': {
            'values': [0.001, 0.005]
        },
        'dropout': {
            'values': [0.2, 0.3]
        },
        'hidden_dim_1': {
            'values': [256, 512]
        },
        'hidden_dim_2': {
            'values': [512]
        },
        'epochs': {
            'value': 3
        }
    }
}

RANDOM_SWEEP_CONFIG = {
    'method': 'random',
    'metric': {
        'name': 'test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [64, 128, 256, 512]
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.02,
            'distribution': 'log_uniform'
        },
        'dropout': {
            'min': 0.1,
            'max': 0.6,
            'distribution': 'uniform'
        },
        'hidden_dim_1': {
            'values': [128, 256, 512, 1024]
        },
        'hidden_dim_2': {
            'values': [256, 512, 1024, 2048]
        },
        'epochs': {
            'value': 5
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
        # Get configuration from wandb
        config = wandb.config
        
        print(f"\n🚀 Starting sweep run with config:")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Dropout: {config.dropout}")
        print(f"  Hidden dims: {config.hidden_dim_1}, {config.hidden_dim_2}")
        print(f"  Epochs: {config.epochs}")
        
        # Create ModelRunSettings from wandb config
        settings = ModelRunSettings(
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            dropout=config.dropout,
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
            continue_model=False  # Don't continue models in sweeps
        )
        
        # Run training with the simplified interface
        results = train_model(settings)
        
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


def create_and_run_sweep(config=None, project_name="hackernews-score-prediction", count=10):
    """
    Create and run a wandb sweep programmatically.
    
    Args:
        config: Sweep configuration dictionary (defaults to SWEEP_CONFIG)
        project_name: W&B project name
        count: Number of runs to execute in the sweep
    """
    if config is None:
        config = SWEEP_CONFIG
    
    print(f"🔧 Creating sweep with {config['method']} optimization...")
    print(f"📊 Target metric: {config['metric']['name']} ({config['metric']['goal']})")
    
    # Create the sweep
    sweep_id = wandb.sweep(config, project=project_name)
    print(f"✅ Sweep created with ID: {sweep_id}")
    print(f"🌐 View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    # Run the sweep
    print(f"🏃 Starting sweep agent with {count} runs...")
    wandb.agent(sweep_id, train_sweep_run, count=count)
    
    print(f"🎉 Sweep completed!")
    return sweep_id


def run_existing_sweep(sweep_id, count=10):
    """
    Run an existing sweep by ID.
    
    Args:
        sweep_id: The ID of an existing sweep
        count: Number of additional runs to execute
    """
    print(f"🔄 Joining existing sweep: {sweep_id}")
    wandb.agent(sweep_id, train_sweep_run, count=count)


def main():
    """
    Main function with different sweep options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter sweeps for HackerNews model')
    parser.add_argument('--config', choices=['default', 'quick', 'random'], default='default',
                        help='Sweep configuration to use (default: default)')
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
    if args.config == 'quick':
        config = QUICK_SWEEP_CONFIG
        print("📋 Using quick grid search configuration")
    elif args.config == 'random':
        config = RANDOM_SWEEP_CONFIG
        print("📋 Using random search configuration")
    else:
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
        run_existing_sweep(args.sweep_id, args.count)
    else:
        sweep_id = create_and_run_sweep(config, args.project, args.count)
        print(f"\n💾 Save this sweep ID for future use: {sweep_id}")


if __name__ == '__main__':
    main()
