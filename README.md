# Contrastive_RL
* This is the implementation of algorithm similar to what is described in the paper Contrastive Unsupervised Representaion for Reinforcement Learning(ICML 2020) [CURL](https://arxiv.org/abs/2004.04136).
* SAC is used along with contratsive losses to speed up the process.
* Tensorboard logs will be uploaded soon.

# Requirements
Anaconda can be installed for package management .
Then create an environment with:
```javascript
conda create -n myenv
source activate myenv
```
Now after entering the environment install the following packages. 

* Python 3.6 or greater will be fine.
* Pytorch 1.14 or greater will be fine.
* gym
* Mujoco
* dmc2gym
* skimage


# Training the model
First adjust the path values and hyper param values in the common file .
```javascript
python train.py

```
And then to see the tensorboard results, in the  command line:
```javascript
python -m tensorboard.main --logdir=[PATH_TO_LOGDIR]
```

The model will be saved in the  specified dir.
