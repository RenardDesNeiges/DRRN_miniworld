## Base policy experiment
**ADAPT NUM_FRAMES SO THAT TRAINING IS APPROX 4H**
`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 5000000 --env-name MiniWorld-FourRooms-v0 --midlevel-rep-names keypoints2d --feature-type base` 

## Midlevel policy experiment


## DRRN policy experiment
**ADAPT NUM_FRAMES SO THAT TRAINING IS APPROX 4H**
`PYOPENGL_PLATFORM=egl python pytorch-a2c-ppo-acktr/main.py --num-frames 5000000 --env-name MiniWorld-FourRooms-v0 --midlevel-rep-names keypoints2d --feature-type drrn --log-interval 1` 