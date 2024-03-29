## Base policy experiment
`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 10000000 --env-name MiniWorld-FourRooms-v0 --feature-type base --lr 0.0000001 --num-processes 32 --num-mini-batch 8` 

## Midlevel policy experiment
`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 2000000 --env-name MiniWorld-FourRooms-v0 --feature-type midlevel_base --log-interval 10 --lr 0.00005`

## DRRN policy experiment
**ADAPT NUM_FRAMES SO THAT TRAINING IS APPROX 4H**
`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 2000000 --env-name MiniWorld-FourRooms-v0 --feature-type drrn --log-interval 10 --lr 0.00005` 
`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 2000000 --env-name MiniWorld-FourRooms-v0 --feature-type drrn --log-interval 10 --lr 0.00001 --save-interval 10` 

`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 2000000 --env-name MiniWorld-FourRooms-v0 --feature-type drrn --log-interval 10 --lr 0.0000001 --save-interval 100 --num-processes 32 --num-mini-batch 8` 
