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

