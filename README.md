# ReinforcementLearning

To run the cartpole example first create a new conda environment and run

```
pip install -e .
python cartpole_pytorch.py
```

Added code using pytorch lightning based on https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/reinforce-learning-DQN.html

To run this do
```
python cartpole_pytorch_lightning.py
```

The design using pytorch lightning is a not completely clear to me as of writing. For one I do not know why there is a target net and how the target for MSE loss is computed. Secondly, because of the way LightningModule naturally works, instead of episodes we have epochs. In the simple pytorch example we would do a new episode when the environment was done and that way we could control number of episodes. Here on the other hand an epoch continues beyond the environment registering a done so the number of episodes is not fixed but rather the global number of training steps.