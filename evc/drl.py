import os
import numpy as np
import math
import matplotlib.pyplot as plt
#import datetime
#import pandas as pd
#import torch as tt
#import torch.nn as nn
#import torch.optim as oo

from .core import ENV, log_evaluations
from .common import now, pj, pjs, dummy, user, REMAP

from stable_baselines3.common.env_checker import check_env 
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


class RLA:
    @staticmethod
    def dqn_01( ds_train, ds_val, ds_test, average_past, lr, lref, batch_size, epochs, nval, result_dir, nfig, do_vis,
                  price_predictor_train,price_predictor_test,price_predictor_val, pp_model_name ):

    #---------------------------------------------
        #pp_model_name = os.path.split(price_predictor.path)[1].split('.')[0]
        dqn_eval_path = pj(f'__models__/rl/dqn_{pp_model_name}')
        dqn_checkpoint_path = pjs(dqn_eval_path,'checkpoints')
        dqn_final_model_path = pjs(dqn_eval_path, 'final_model')
        dqn_best_model_path = pjs(dqn_eval_path, 'best_model')
        #---------------------------------------------
    # # DQN Training

    # ## Training Environments
        if ds_train is not None:
    
            env = ENV(  price_dataset=ds_train, price_predictor=price_predictor_train,
                    average_past = average_past, random_days=True,  frozen=False, seed=12 )
            venv = ENV( price_dataset=ds_val,  price_predictor=price_predictor_val,
                    average_past = average_past, random_days=False, frozen=True,  seed=13 )

            # check_env(env) #<--- optinally check
            # check_env(venv) #<--- optinally check
            print(f'{env=}, {env.n_days=}')
            print(f'{venv=}, {venv.n_days=}')
                

            ### Build DQN
           
            # learning rate scheduling
            start_lr, end_lr = float(lr), float(lr)*float(lref)      # select-arg
            lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
            def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr
            
            
            '''
            start_lr = 3e-4
            k = 8e-6
            t = int(epochs)
            def lr_schedule(progress): return start_lr*math.exp(-k*t*(1-progress))
            '''
            
            model = DQN(
            policy =           'MlpPolicy',
            env =               env,
            learning_rate=      lr_schedule,
            buffer_size=        1_000_000,
            learning_starts=    5_000,
            batch_size=         int(batch_size),
            tau=                1,
            gamma=              0.95,
            train_freq=         20,
            gradient_steps=            50,
            target_update_interval=    20,
            exploration_fraction=      0.1,
            exploration_initial_eps=   1.0,
            exploration_final_eps=     0.1,
            verbose=                   1,
            device=                    'cpu',
            policy_kwargs=dict(
                                    #activation_fn=  nn.LeakyReLU,
                                    net_arch=[400, 300, 300]))
            print(model)

            # ## Train DQN

        
            total_timesteps=   int(epochs)
            log_interval =     int(total_timesteps/10) #<--- enable verbose to get actual log
            #---------------------------------------------
            eval_freq=         int(total_timesteps/int(nval)) # evaluate times
            save_freq=         int(total_timesteps/10) # checkpoint times

            #---------------------------------------------
            start_time=now()
            print('Training DQN', start_time)
            #---------------------------------------------
            eval_callback = EvalCallback(venv, 
                best_model_save_path  =  dqn_eval_path,
                log_path =               dqn_eval_path, 
                eval_freq =              eval_freq, 
                n_eval_episodes =        venv.day_count)

            checkpoint_callback = CheckpointCallback(
                save_freq=               save_freq, 
                save_path=               dqn_checkpoint_path)
            #---------------------------------------------

            model.learn(
                total_timesteps=total_timesteps,
                log_interval=log_interval, #int(0.1*total_timesteps)
                progress_bar = True,
                callback = CallbackList([checkpoint_callback, eval_callback]) # Create the callback list
                #cb = lambda l, g : print('callback', now(), '\n', l, '\n', g)
            )
            model.save(dqn_final_model_path)
            #---------------------------------------------

            print('Saved DQN @ ', dqn_final_model_path)
            end_time=now()
            print('Time Elapsed: {}'.format(end_time-start_time))

            # ### Validation Results


            #log_evaluations(os.path.join(dqn_eval_path,'evaluations.npz'))

            #print(f'\n{dqn_eval_path=}')
            #print(f'{dqn_final_model_path=}')
            #print(f'{dqn_best_model_path=}')


        if ds_test is not None:
        # # DQN Testing

            # ## Testing Environment


            tenv = ENV(
                price_dataset=    ds_test,         
                price_predictor=  price_predictor_test,
                average_past=     average_past,
                random_days=      False,           
                frozen =          True,
                seed=             14,
                record =          True
            )
            print(tenv, tenv.n_days)


            # ## Select DQN


            model = DQN.load(dqn_best_model_path)
            #model = DQN.load(dqn_final_model_path)


            # #Perform Testing [DQN]

        
            #history, mean_episode_return, sum_episode_cost, total_episodes = \
            #    tenv.test(model, verbose=1, render=True, rmode=0, plot=True)
            result_dir = f'{result_dir}_dqn_new'
            os.makedirs(result_dir, exist_ok=True)
            history, mean_episode_return, sum_episode_cost, total_episodes, episode_nos, episode_returns, episode_costs = \
                tenv.test(model, pp_model_name, verbose=0, render=False, rmode=0, plot=True, result_dir=result_dir)

            print(f'{mean_episode_return=}, {sum_episode_cost=}, {total_episodes=}')
  
            
            #xplot = np.arange(len(episode_returns))
            fig = plt.figure(figsize=(len(episode_returns)*0.2,8))
            plt.bar(episode_nos, episode_returns, color='tab:blue')
            plt.xlabel('episode')
            plt.ylabel('return')
            plt.grid(axis='both')
            plt.xticks(episode_nos, fontsize=8, rotation=90, labels=episode_nos+1)
            fig.savefig(pjs(result_dir, f'test_return_{pp_model_name}.png'))

            fig = plt.figure(figsize=(len(episode_costs)*0.2,8))
            plt.bar(episode_nos, episode_costs, color='tab:green')
            plt.xlabel('episode')
            plt.ylabel('cost')
            plt.grid(axis='both')
            plt.xticks(episode_nos, fontsize=8, rotation=90, labels=episode_nos+1)
            fig.savefig(pjs(result_dir, f'test_cost_{pp_model_name}.png'))

            # ### Segmented Plot [DQN]

            #NOTE: use "fix_ylim=0" to not fix the y-axis limits
            #NOTE: use "legend=True" to show legend
            if do_vis:
                figures = tenv.segment_plot(duration_per_graph=75, history=history, 
                                fix_ylim=25, max_segments=nfig, legend=False, 
                                episode_index=True, grid=False, price_scale=0.5, cost_scale=0.05)
                for x,y,fig in figures:
                    fig.savefig(pjs(result_dir, f'test_vis_{x}_{y}_{pp_model_name}.png'))

    @staticmethod
    def ddpg_01( ds_train, ds_val, ds_test, average_past, lr, lref, batch_size, epochs, nval, result_dir, nfig, do_vis,
                   price_predictor_train,price_predictor_test,price_predictor_val,pp_model_name):

        #---------------------------------------------
        #---------------------------------------------
        #pp_model_name = os.path.split(price_predictor.path)[1].split('.')[0]
        ddpg_eval_path = pj(f'__models__/rl/ddpg_{pp_model_name}')
        ddpg_checkpoint_path = pjs(ddpg_eval_path,'checkpoints')
        ddpg_final_model_path = pjs(ddpg_eval_path, 'final_model')
        ddpg_best_model_path = pjs(ddpg_eval_path, 'best_model')
        #---------------------------------------------

    # ## Training Environments
        if ds_train is not None:
            print("DOING TRAINING: ###########################")
    
            env = ENV(  price_dataset=ds_train, price_predictor=price_predictor_train,
                    average_past = average_past, random_days=True,  frozen=False, discrete=False, seed=12 )
            venv = ENV( price_dataset=ds_val,  price_predictor=price_predictor_val,
                    average_past = average_past, random_days=False, frozen=True, discrete=False, seed=13 )

            # check_env(env) #<--- optinally check
            # check_env(venv) #<--- optinally check
            print(f'{env=}, {env.n_days=}')
            print(f'{venv=}, {venv.n_days=}')
                

            # ## Build

                # learning rate scheduling
            
          
            start_lr, end_lr = float(lr), (float(lr))*(float(lref))      # select-arg
            lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
            def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr
            '''
            start_lr = 3e-3
            k = 8e-6
            t = int(epochs)
            def lr_schedule(progress): return start_lr*math.exp(-k*t*(1-progress))
            '''
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(env.action_space.shape, dtype=env.action_space.dtype), 
                sigma=np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + 1.0, 
                theta=0.15, dt=0.01, initial_noise=None)
            model = DDPG(
            policy =              'MlpPolicy', 
            env =                 env, 
            learning_rate=        lr_schedule,
            buffer_size=          1_000_000,
            learning_starts=      5_000,
            batch_size=           int(batch_size),
            tau=                  1,
            gamma=                0.95,
            action_noise=         action_noise,
            train_freq=           (10, 'episode'),
            gradient_steps=            50,
            #target_update_interval=    50,
            verbose=                   1,
            device=                    'cpu',
            policy_kwargs=dict(
                                    #activation_fn=  nn.LeakyReLU, 
                                    net_arch=[400, 300, 300]))
            print(model)
            # ## Train DQN

        
            total_timesteps=   int(epochs)
            log_interval =     int(total_timesteps/10) #<--- enable verbose to get actual log
            #---------------------------------------------
            eval_freq=         int(total_timesteps/int(nval)) # evaluate times
            save_freq=         int(total_timesteps/10) # checkpoint times

            #---------------------------------------------
            start_time=now()
            print('Training DDPG', start_time)
            #---------------------------------------------
            eval_callback = EvalCallback(venv, 
                best_model_save_path  =  ddpg_eval_path,
                log_path =               ddpg_eval_path, 
                eval_freq =              eval_freq, 
                n_eval_episodes =        venv.day_count)

            checkpoint_callback = CheckpointCallback(
                save_freq=               save_freq, 
                save_path=               ddpg_checkpoint_path)
            #---------------------------------------------

            model.learn(
                total_timesteps=total_timesteps,
                progress_bar = True,
                log_interval=log_interval, #int(0.1*total_timesteps)
                callback = CallbackList([checkpoint_callback, eval_callback]) # Create the callback list
                #cb = lambda l, g : print('callback', now(), '\n', l, '\n', g)
            )
            model.save(ddpg_final_model_path)
            #---------------------------------------------

            print('Saved DDPG @ ', ddpg_final_model_path)
            end_time=now()
            print('Time Elapsed: {}'.format(end_time-start_time))

            # ### Validation Results


            #log_evaluations(os.path.join(dqn_eval_path,'evaluations.npz'))

            #print(f'\n{dqn_eval_path=}')
            #print(f'{dqn_final_model_path=}')
            #print(f'{dqn_best_model_path=}')


        if ds_test is not None:
        # # Testing

            # ## Testing Environment


            tenv = ENV(
                price_dataset=    ds_test,         
                price_predictor=  price_predictor_test,
                average_past=     average_past,
                random_days=      False,           
                frozen =          True,
                seed=             14,
                discrete=         False,
                record =          True
            )
            print(tenv, tenv.n_days)


            # ## Select DQN


            model = DDPG.load(ddpg_best_model_path)
            #model = DDPG.load(ddpg_final_model_path)


            # #Perform Testing [DQN]

        
            #history, mean_episode_return, sum_episode_cost, total_episodes = \
            #    tenv.test(model, verbose=1, render=True, rmode=0, plot=True)
            
            result_dir = f'{result_dir}_ddpg_new'
            os.makedirs(result_dir, exist_ok=True)
            
            history, mean_episode_return, sum_episode_cost, total_episodes, episode_nos, episode_returns, episode_costs = \
                tenv.test(model, pp_model_name, verbose=0, render=False, rmode=0, plot=True, result_dir=result_dir)

            print(f'{mean_episode_return=}, {sum_episode_cost=}, {total_episodes=}')
            #if plot:
            
            #xplot = np.arange(len(episode_returns))
            fig = plt.figure(figsize=(len(episode_returns)*0.2,8))
            plt.bar(episode_nos, episode_returns, color='tab:blue')
            plt.xlabel('episode')
            plt.ylabel('return')
            plt.grid(axis='both')
            plt.xticks(episode_nos, fontsize=8, rotation=90, labels=episode_nos+1)
            fig.savefig(pjs(result_dir, f'test_return_{pp_model_name}.png'))

            fig = plt.figure(figsize=(len(episode_costs)*0.2,8))
            plt.bar(episode_nos, episode_costs, color='tab:green')
            plt.xlabel('episode')
            plt.ylabel('cost')
            plt.grid(axis='both')
            plt.xticks(episode_nos, fontsize=8, rotation=90, labels=episode_nos+1)
            fig.savefig(pjs(result_dir, f'test_cost_{pp_model_name}.png'))

            # ### Segmented Plot [DQN]

            #NOTE: use "fix_ylim=0" to not fix the y-axis limits
            #NOTE: use "legend=True" to show legend
            if do_vis:
                figures = tenv.segment_plot(duration_per_graph=75, history=history, 
                                fix_ylim=25, max_segments=nfig, legend=False, 
                                episode_index=True, grid=False, price_scale=0.5, cost_scale=0.05)
                for x,y,fig in figures:
                    fig.savefig(pjs(result_dir, f'test_vis_{x}_{y}_{pp_model_name}.png'))


    @staticmethod
    def ppo_disc_01( ds_train, ds_val, ds_test, average_past, lr, lref, batch_size, epochs, nval, result_dir, nfig, do_vis,
                   price_predictor_train,price_predictor_test,price_predictor_val,pp_model_name):

        #---------------------------------------------
        #---------------------------------------------
        #pp_model_name = os.path.split(price_predictor.path)[1].split('.')[0]
        ppo_disc_eval_path = pj(f'__models__/rl/ppo_disc_{pp_model_name}')
        ppo_disc_checkpoint_path = pjs(ppo_disc_eval_path,'checkpoints')
        ppo_disc_final_model_path = pjs(ppo_disc_eval_path, 'final_model')
        ppo_disc_best_model_path = pjs(ppo_disc_eval_path, 'best_model')
        #---------------------------------------------

    # ## Training Environments
        if ds_train is not None:
    
            env = ENV(  price_dataset=ds_train, price_predictor=price_predictor_train,
                    average_past = average_past, random_days=True,  frozen=False, discrete=True, seed=12 )
            venv = ENV( price_dataset=ds_val,  price_predictor=price_predictor_val,
                    average_past = average_past, random_days=False, frozen=True, discrete=True, seed=13 )

            # check_env(env) #<--- optinally check
            # check_env(venv) #<--- optinally check
            print(f'{env=}, {env.n_days=}')
            print(f'{venv=}, {venv.n_days=}')
                

            # ## Build
            '''
            # learning rate scheduling
            start_lr, end_lr = float(lr)+0.00001, (float(lr)+0.00001)*(float(lref)+0.35)      # select-arg #bestttt
            lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
            def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr
            
            '''
            start_lr, end_lr = float(lr)+0.0001, (float(lr)+0.0001)*(float(lref)+0.15)      # select-arg
            lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
            def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr
            

            start_lr = 5e-4
            k = 3.33e-6
            t = int(epochs)
            def lr_schedule(progress): return start_lr*math.exp(-k*t*(1-progress))

            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(env.action_space.shape, dtype=env.action_space.dtype), 
                sigma=np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + 1.0, 
                theta=0.15, dt=0.01, initial_noise=None)
            model = PPO(
            policy =              'MlpPolicy', 
            env =                 env, 
            learning_rate=        lr_schedule,
            batch_size=           int(batch_size),  
            gamma=                0.95,
            verbose=                   1,
            device=                    'cpu',
            n_steps = 2048,
            n_epochs = 25,
            clip_range = 0.2
            )
            print(model)
            # ## Train DQN

        
            total_timesteps=   int(epochs)
            log_interval =     int(total_timesteps/10) #<--- enable verbose to get actual log
            #---------------------------------------------
            eval_freq=         int(total_timesteps/int(nval)) # evaluate times
            save_freq=         int(total_timesteps/10) # checkpoint times

            #---------------------------------------------
            start_time=now()
            print('Training PPO with discrete actions', start_time)
            #---------------------------------------------
            eval_callback = EvalCallback(venv, 
                best_model_save_path  =  ppo_disc_eval_path,
                log_path =               ppo_disc_eval_path, 
                eval_freq =              eval_freq, 
                n_eval_episodes =        venv.day_count)

            checkpoint_callback = CheckpointCallback(
                save_freq=               save_freq, 
                save_path=               ppo_disc_checkpoint_path)
            #---------------------------------------------

            model.learn(
                total_timesteps=total_timesteps,
                progress_bar = True,
                log_interval=log_interval, #int(0.1*total_timesteps)
                callback = CallbackList([checkpoint_callback, eval_callback]) # Create the callback list
                #cb = lambda l, g : print('callback', now(), '\n', l, '\n', g)
            )
            model.save(ppo_disc_final_model_path)
            #---------------------------------------------

            print('Saved PPO_discrete @ ', ppo_disc_final_model_path)
            end_time=now()
            print('Time Elapsed: {}'.format(end_time-start_time))

            # ### Validation Results


            #log_evaluations(os.path.join(dqn_eval_path,'evaluations.npz'))

            #print(f'\n{dqn_eval_path=}')
            #print(f'{dqn_final_model_path=}')
            #print(f'{dqn_best_model_path=}')


        if ds_test is not None:
        # # Testing

            # ## Testing Environment


            tenv = ENV(
                price_dataset=    ds_test,         
                price_predictor=  price_predictor_test,
                average_past=     average_past,
                random_days=      False,           
                frozen =          True,
                seed=             14,
                discrete=         True,
                record =          True
            )
            print(tenv, tenv.n_days)


            # ## Select DQN


            model = PPO.load(ppo_disc_best_model_path)
            #model = DDPG.load(ddpg_final_model_path)


            # #Perform Testing [DQN]

        
            #history, mean_episode_return, sum_episode_cost, total_episodes = \
            #    tenv.test(model, verbose=1, render=True, rmode=0, plot=True)
            
            result_dir = f'{result_dir}_ppo_disc_01_new'
            os.makedirs(result_dir, exist_ok=True)
            
            history, mean_episode_return, sum_episode_cost, total_episodes, episode_nos, episode_returns, episode_costs = \
                tenv.test(model, pp_model_name, verbose=0, render=False, rmode=0, plot=True, result_dir=result_dir)

            print(f'{mean_episode_return=}, {sum_episode_cost=}, {total_episodes=}')
            #if plot:
            
            #xplot = np.arange(len(episode_returns))
            fig = plt.figure(figsize=(len(episode_returns)*0.2,8))
            plt.bar(episode_nos, episode_returns, color='tab:blue')
            plt.xlabel('episode')
            plt.ylabel('return')
            plt.grid(axis='both')
            plt.xticks(episode_nos, fontsize=8, rotation=90, labels=episode_nos+1)
            fig.savefig(pjs(result_dir, f'test_return_{pp_model_name}.png'))
            plt.clf()
            plt.close()

            fig = plt.figure(figsize=(len(episode_costs)*0.2,8))
            plt.bar(episode_nos, episode_costs, color='tab:green')
            plt.xlabel('episode')
            plt.ylabel('cost')
            plt.grid(axis='both')
            plt.xticks(episode_nos, fontsize=8, rotation=90, labels=episode_nos+1)
            fig.savefig(pjs(result_dir, f'test_cost_{pp_model_name}.png'))
            plt.clf()
            plt.close()
            # ### Segmented Plot [DQN]

            #NOTE: use "fix_ylim=0" to not fix the y-axis limits
            #NOTE: use "legend=True" to show legend
            if do_vis:
                figures = tenv.segment_plot(duration_per_graph=75, history=history, 
                                fix_ylim=25, max_segments=nfig, legend=False, 
                                episode_index=True, grid=False, price_scale=0.5, cost_scale=0.05)
                for x,y,fig in figures:
                    fig.savefig(pjs(result_dir, f'test_vis_{x}_{y}_{pp_model_name}.png'))
                    plt.clf()
                    plt.close()


    @staticmethod
    def ppo_cont_01( ds_train, ds_val, ds_test, average_past, lr, lref, batch_size, epochs, nval, result_dir, nfig, do_vis,
                   price_predictor_train,price_predictor_test,price_predictor_val,pp_model_name):

        #---------------------------------------------
        #---------------------------------------------
        #pp_model_name = os.path.split(price_predictor.path)[1].split('.')[0]
        ppo_cont_eval_path = pj(f'__models__/rl/ppo_cont_{pp_model_name}')
        ppo_cont_checkpoint_path = pjs(ppo_cont_eval_path,'checkpoints')
        ppo_cont_final_model_path = pjs(ppo_cont_eval_path, 'final_model')
        ppo_cont_best_model_path = pjs(ppo_cont_eval_path, 'best_model')
        #---------------------------------------------

    # ## Training Environments
        if ds_train is not None:
    
            env = ENV(  price_dataset=ds_train, price_predictor=price_predictor_train,
                    average_past = average_past, random_days=True,  frozen=False, discrete=False, seed=12 )
            venv = ENV( price_dataset=ds_val,  price_predictor=price_predictor_val,
                    average_past = average_past, random_days=False, frozen=True, discrete=False, seed=13 )

            # check_env(env) #<--- optinally check
            # check_env(venv) #<--- optinally check
            print(f'{env=}, {env.n_days=}')
            print(f'{venv=}, {venv.n_days=}')
                

            # ## Build

                # learning rate scheduling
            
            start_lr, end_lr = float(lr)+0.00001, (float(lr)+0.00001)*(float(lref)+0.35)      # select-arg
            lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
            def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr
            
            start_lr = 5e-4
            k = 3.33e-6
            t = int(epochs)
            def lr_schedule(progress): return start_lr*math.exp(-k*t*(1-progress))

            model = PPO(
            policy =              'MlpPolicy', 
            env =                 env, 
            learning_rate=        lr_schedule,
            batch_size=           int(batch_size),  
            gamma=                0.95,
            verbose=                   1,
            n_epochs = 30,
            n_steps = 256,
            device=               'cpu')

            '''
            start_lr, end_lr = float(lr)+0.0001, (float(lr)+0.0001)*(float(lref)+0.15)      # select-arg
            lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer            print(model)
            # ## Train DQN
            '''
        
            total_timesteps=   int(epochs)
            log_interval =     int(total_timesteps/10) #<--- enable verbose to get actual log
            #---------------------------------------------
            eval_freq=         int(total_timesteps/int(nval)) # evaluate times
            save_freq=         int(total_timesteps/10) # checkpoint times

            #---------------------------------------------
            start_time=now()
            print('Training PPO with continuous actions ', start_time)
            #---------------------------------------------
            eval_callback = EvalCallback(venv, 
                best_model_save_path  =  ppo_cont_eval_path,
                log_path =               ppo_cont_eval_path, 
                eval_freq =              eval_freq, 
                n_eval_episodes =        venv.day_count)

            checkpoint_callback = CheckpointCallback(
                save_freq=               save_freq, 
                save_path=               ppo_cont_checkpoint_path)
            #---------------------------------------------

            model.learn(
                total_timesteps=total_timesteps,
                progress_bar = True,
                log_interval=log_interval, #int(0.1*total_timesteps)
                callback = CallbackList([checkpoint_callback, eval_callback]) # Create the callback list
                #cb = lambda l, g : print('callback', now(), '\n', l, '\n', g)
            )
            model.save(ppo_cont_final_model_path)
            #---------------------------------------------

            print('Saved PPO_continuous @ ', ppo_cont_final_model_path)
            end_time=now()
            print('Time Elapsed: {}'.format(end_time-start_time))

            # ### Validation Results


            #log_evaluations(os.path.join(dqn_eval_path,'evaluations.npz'))

            #print(f'\n{dqn_eval_path=}')
            #print(f'{dqn_final_model_path=}')
            #print(f'{dqn_best_model_path=}')


        if ds_test is not None:
        # # Testing

            # ## Testing Environment


            tenv = ENV(
                price_dataset=    ds_test,         
                price_predictor=  price_predictor_test,
                average_past=     average_past,
                random_days=      False,           
                frozen =          True,
                seed=             14,
                discrete=         False,
                record =          True
            )
            print(tenv, tenv.n_days)


            # ## Select DQN


            model = PPO.load(ppo_cont_best_model_path)
            #model = DDPG.load(ddpg_final_model_path)


            # #Perform Testing [DQN]

        
            #history, mean_episode_return, sum_episode_cost, total_episodes = \
            #    tenv.test(model, verbose=1, render=True, rmode=0, plot=True)
            
            result_dir = f'{result_dir}_ppo_cont_01_new'
            os.makedirs(result_dir, exist_ok=True)
            
            history, mean_episode_return, sum_episode_cost, total_episodes, episode_nos, episode_returns, episode_costs = \
                tenv.test(model, pp_model_name, verbose=0, render=False, rmode=0, plot=True, result_dir=result_dir)

            print(f'{mean_episode_return=}, {sum_episode_cost=}, {total_episodes=}')
            #if plot:
            #xplot = np.arange(len(episode_returns))
            fig = plt.figure(figsize=(len(episode_returns)*0.2,8))
            plt.bar(episode_nos, episode_returns, color='tab:blue')
            plt.xlabel('episode')
            plt.ylabel('return')
            plt.grid(axis='both')
            plt.xticks(episode_nos, fontsize=8, rotation=90, labels=episode_nos+1)
            fig.savefig(pjs(result_dir, f'test_return_{pp_model_name}.png'))

            fig = plt.figure(figsize=(len(episode_costs)*0.2,8))
            plt.bar(episode_nos, episode_costs, color='tab:green')
            plt.xlabel('episode')
            plt.ylabel('cost')
            plt.grid(axis='both')
            plt.xticks(episode_nos, fontsize=8, rotation=90, labels=episode_nos+1)
            fig.savefig(pjs(result_dir, f'test_cost_{pp_model_name}.png'))

            # ### Segmented Plot [DQN]

            #NOTE: use "fix_ylim=0" to not fix the y-axis limits
            #NOTE: use "legend=True" to show legend
            if do_vis:
                figures = tenv.segment_plot(duration_per_graph=75, history=history, 
                                fix_ylim=25, max_segments=nfig, legend=False, 
                                episode_index=True, grid=False, price_scale=0.5, cost_scale=0.05)
                for x,y,fig in figures:
                    fig.savefig(pjs(result_dir, f'test_vis_{x}_{y}_{pp_model_name}.png'))

 
