## Base policy experiment
**ADAPT NUM_FRAMES SO THAT TRAINING IS APPROX 4H**
`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 5000000 --env-name MiniWorld-FourRooms-v0 --midlevel-rep-names keypoints2d --feature-type base` 

## Midlevel policy experiment
`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 5000000 --env-name MiniWorld-FourRooms-v0 --midlevel-rep-names keypoints2d --feature-type midlevel_base --log-interval 1`

## DRRN policy experiment
**ADAPT NUM_FRAMES SO THAT TRAINING IS APPROX 4H**
`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 2000000 --env-name MiniWorld-FourRooms-v0 --feature-type drrn --log-interval 10 --lr 0.00005` 