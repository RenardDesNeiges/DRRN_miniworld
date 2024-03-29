import copy
import glob
import os
import time
import types
from collections import deque

import numpy as np
import torch

import algo
from arguments import get_args
from envs import make_vec_envs
from policy import Policy
from storage import RolloutStorage
from create_logger import create_logger,Logger_tensorboard,make_path
import datetime
import logging

args = get_args()

if args.feature_type == 'drrn':
    recurrent_policy = True
else:
    recurrent_policy = False

assert args.algo == 'ppo', 'Unsupported policy specified'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

tf_dir =os.path.normpath(
    make_path(os.path.join("trained_models", args.feature_type+datetime.datetime.now().strftime("_%m_%d_%H_%M"))))
_ = create_logger(tf_dir)
logger_tb = Logger_tensorboard(tf_dir, use_tensorboard=True)

eval_log_dir = tf_dir + "_eval"


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    """
    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
    """

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, tf_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                          network=args.feature_type,
                          base_kwargs={
                              'recurrent': recurrent_policy,
                              'mid_level_reps':args.midlevel_rep_names
                          })

    actor_critic.to(device)

    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    value_losses = deque(maxlen=100)
    action_losses = deque(maxlen=100)

    start = time.time()
    for j in range(num_updates):
        rew_tot = 0
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            rew_tot += reward
            """
            for info in infos:
                if 'episode' in info.keys():
                    print(reward)
                    episode_rewards.append(info['episode']['r'])
            """
            
            # FIXME: works only for environments with sparse rewards
            for idx, eps_done in enumerate(done):
                
                if eps_done:
                    episode_rewards.append(rew_tot[idx]/step)



            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        value_losses.append(value_loss)
        action_losses.append(action_loss)
        rollouts.after_update()

        if j % args.save_interval == 0:
            print('Saving model')

            save_path = tf_dir
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]
            torch.save(save_model, os.path.join(save_path, args.env_name+"_step_"+str(j) + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()

            if type(episode_rewards[-1]) == torch.Tensor:
                episode_rewards = [float(ep) for ep in episode_rewards]

            message="Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, " \
                    "min/max reward {:.2f}/{:.2f}, success rate {:.2f}, value_loss{:4f},action_loss{:4f} \n".format(
                    j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    np.count_nonzero(np.greater(episode_rewards, 0.1)) / len(episode_rewards),
                    np.mean(value_losses),
                    np.mean(action_losses),
                )
            logging.info(message)
            print(message)
            logger_tb.add_losses({'mean reward ': np.mean(episode_rewards),
                                  " success rate":np.count_nonzero(np.greater(episode_rewards, 0.1)) / len(episode_rewards)
                                  }, total_num_steps)

        if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
            eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                                      args.gamma, eval_log_dir, args.add_timestep, device, True)

            if eval_envs.venv.__class__.__name__ == "VecNormalize":
                eval_envs.venv.ob_rms = envs.venv.ob_rms

                # An ugly hack to remove updates
                def _obfilt(self, obs):
                    if self.ob_rms:
                        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob,
                                      self.clipob)
                        return obs
                    else:
                        return obs

                eval_envs.venv._obfilt = types.MethodType(_obfilt, envs.venv)

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                                       actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            message=" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(eval_episode_rewards),
                np.mean(eval_episode_rewards)
            )
            logging.info(message)
            print(message)
            logger_tb.add_losses({'eval mean reward ': np.mean(eval_episode_rewards)})

        """
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass
        """

    envs.close()


if __name__ == "__main__":
    main()
