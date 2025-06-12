# Hyperparameter Sweeping Guide

This directory now contains both CLI-based and programmatic approaches to hyperparameter sweeping with Weights & Biases.

## 📈 Monitoring

View your sweeps in the Weights & Biases dashboard:
- Go to https://wandb.ai/YOUR_USERNAME/hackernews-score-prediction
- Navigate to "Sweeps" tab
- Monitor progress, compare runs, and analyze results

## 🎛️ Customizing Sweeps

To create custom sweep configurations, edit the configuration dictionaries in `sweep.py`:

```python
CUSTOM_SWEEP_CONFIG = {
    'method': 'bayes',
    'metric': {
        'name': 'test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.1,
            'distribution': 'log_uniform'
        },
        # Add your parameters here
    }
}
```

## 📝 Example Workflow

```bash
# 1. Run small sweep to verify everything works
uv run ./model/sweep.py --config quick --count 5

# 2. Run full optimization sweep
uv run ./model/sweep.py --count 50

# 3. Continue with more runs if needed
uv run ./model/sweep.py --sweep-id SWEEP_ID_FROM_STEP_2 --count 25
```
