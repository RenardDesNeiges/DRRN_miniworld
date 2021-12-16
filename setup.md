# commands to run to install the env

```
conda create --name miniworld python=3.8
conda activate miniworld

conda install -c conda-forge gym

conda install numpy 

pip install pyglet --user

pip install -e .
```

to run anything on a remote server run : `PYOPENGL_PLATFORM=egl benchmark.py `.  *(to enable running without a display)*



example training : `PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --algo ppo --num-frames 5000000 --num-processes 16 --num-steps 80 --lr 0.00005 --env-name MiniWorld-Hallway-v0 `.  *(to enable running without a display)*

