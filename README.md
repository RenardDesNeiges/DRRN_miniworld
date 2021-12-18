# Deep Robust Drone Navigation With Deep Cognitive Mapping and Visual Priors

Vision based RL agent for object navigation using deep cognitive mapping(see, [Sax et al. 2019](https://arxiv.org/abs/1702.03920)) and visual priors (see, [Gupta et al. 2017](https://arxiv.org/abs/1912.11121)).

The simulation environment is [gym-miniworld](https://github.com/maximecb/gym-miniworld). The gym-miniworld available environments [are listed here](docs/environments.md),  [a design tutorial](docs/design.md) is also available here, [as well as a troubleshooting guide](docs/troubleshooting.md).

The RL environment is [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).

Our experiments are listed (and the exact arguments specified) [here](docs/experiments.md).

This is a project for the course *"Visual Intelligence, Machines and Minds"* (CS-503) at EPFL. By the team of Umer Hasan, Yongtao Wu and Titouan Renard. 


## Setup

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- Pyglet (OpenGL 3D graphics)
- pytorch, torchvision
- tensorboard
- matplotlib
- visualpriors
- GPU for 3D graphics acceleration (optional)

We recommend using conda to manage the environment (the installation is still a bit messy).

```
conda create --name cuda_miniworld python=3.9.7
conda activate cuda_world
pip install pyglet --user
python setup.py install
pip install -e .
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install matplotlib
conda install tensorboard
pip install visualpriors
```

## Usage

### Testing the environments

```
./manual_control.py --env-name MiniWorld-Hallway-v0

# Display an overhead view of the environment
./manual_control.py --env-name MiniWorld-Hallway-v0 --top_view
```

There is also a script to run automated tests (`run_tests.py`) and a script to gather performance metrics (`benchmark.py`).

### Offscreen Rendering

You can run `gym-miniword` offscreen by setting the environment variable `PYOPENGL_PLATFORM` to `egl` before running MiniWorld, e.g.

```
PYOPENGL_PLATFORM=egl python3 main.py --algo ppo --num-frames 5000000 --num-processes 16 --num-steps 80 --lr 0.00005 --env-name MiniWorld-Hallway-v0
```

### Training the models

To train the agents use :

```
PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 5000000 --env-name MiniWorld-FourRooms-v0 --midlevel-rep-names keypoints2d --feature-type drrn --log-interval 1 --num-steps 80 --lr 0.00005
```

Then, to visualize the results of training :

```
python3 pytorch-a2c-ppo-acktr/enjoy.py --env-name MiniWorld-Hallway-v0 --load-path trained_models/base/2021XXXX/MiniWorld-Hallway-v0-stepX.pt
```

