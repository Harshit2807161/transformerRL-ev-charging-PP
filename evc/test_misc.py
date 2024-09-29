def test_skip(self, model, episodes, verbose=1, render=True, rmode=0, plot=True):

    # When performing this test, 
    # restart the environemnt and make sure random_days is false
    
    self.restart() # <--restart once
    max_episodes = self.day_count
    print(f'Testing Environment: [{self.price_dataset.csv}] : for [{max_episodes=}]')

    if not self.frozen: print('WARNING:: recomended using [frozen = True]')
    if self.random_days: print('WARNING:: recomended using [random_days = False]')
    
    print(f'{self.frozen=}, {self.random_days=}\n')
    
    #-------------------
    episode = 0
    episode_returns = []
    episode_nos = []
    episode_costs= []
    while episode<max_episodes:
        episode+=1
        obs = self.reset()
        if episode not in episodes:continue
            
        episode_return = 0
        episode_cost=0
        episode_timesteps, done = 0, False
        episode_nos.append(episode)
        if verbose>0: 
            print('=================================================================')
            print(f'[{episode=}] :: {self.arrival_time=}, {self.departure_time=}')
        if render: self.render(mode=rmode)
        while not done:
            episode_timesteps+=1
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = self.step(act)
            episode_return+=rew
            step_cost=info['step_cost']
            episode_cost+=step_cost
            
            if verbose>0: 
                print(f'--> [t={episode_timesteps}] :: [{act=}:->{self.action_mapper(act)}] : {rew=:.2f} ~{done=} ~{step_cost= }')
            if render: self.render(mode=rmode)
        if verbose>0:
            print(f'\nDone: {episode_timesteps=}, {episode_return=}, {episode_cost=}')
            print('=================================================================')

        episode_timesteps+=1
        episode_returns.append(episode_return)
        episode_costs.append(episode_cost)

    assert(not self.drem)
    if plot:
        xplot = np.arange(len(episode_returns))
        plt.figure(figsize=(len(episode_returns)*0.2,8))
        plt.bar(xplot, episode_returns)
        plt.xlabel('episode')
        plt.ylabel('return')
        plt.grid(axis='both')
        plt.xticks(xplot, fontsize=8, rotation=90, labels=episode_nos)
        #plt.show()

        plt.figure(figsize=(len(episode_costs)*0.2,8))
        plt.bar(xplot, episode_costs, color='tab:green')
        plt.xlabel('episode')
        plt.ylabel('cost')
        plt.grid(axis='both')
        plt.xticks(xplot, fontsize=8, rotation=90, labels=xplot+1)
        #plt.show()

    mean_episode_return = np.mean(episode_returns)
    sum_episode_cost = np.sum(episode_costs)

    total_episodes = len(episode_returns)
    print(f'Avg Episode Return: {mean_episode_return} over [{total_episodes}] episodes')
    print(f'Total Episode Cost: {sum_episode_cost} over [{total_episodes}] episodes')
    #print(f'cost : {total_cost} over [{total_episodes}] episodes')
    
    return np.array(self.history), mean_episode_return, sum_episode_cost, total_episodes

def test_user(self, default_model, user_model, user_episodes, user_verbose=1, user_render=True, rmode=0, plot=True):

    # When performing this test, 
    # restart the environemnt and make sure random_days is false
    
    self.restart() # <--restart once
    max_episodes = self.day_count
    print(f'Manual Testing Environment: [{self.price_dataset.csv}] : for [{max_episodes=}]')

    if not self.frozen: print('WARNING:: recomended using [frozen = True]')
    if self.random_days: print('WARNING:: recomended using [random_days = False]')
    
    print(f'{self.frozen=}, {self.random_days=}\n')
    
    #-------------------
    episode = 0
    episode_returns = []
    episode_costs= []
    while episode<max_episodes:
        episode+=1
        episode_return = 0
        episode_cost=0
        episode_timesteps, done = 0, False
        obs = self.reset()
        if episode in user_episodes:
            model = user_model
            verbose=user_verbose
            render=user_render
        else:
            model = default_model
            verbose=0
            render=False
            
        if verbose>0: 
            print('=================================================================')
            print(f'[{episode=}] :: {self.arrival_time=}, {self.departure_time=}')
        if render: self.render(mode=rmode)

        while not done:
            episode_timesteps+=1
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = self.step(act)
            episode_return+=rew
            step_cost=info['step_cost']
            episode_cost+=step_cost
            
            if verbose>0: 
                print(f'--> [t={episode_timesteps}] :: [{act=}:->{self.action_mapper(act)}] : {rew=:.2f} ~{done=} ~{step_cost= }')
            if render: self.render(mode=rmode)
        if verbose>0:
            print(f'\nDone: {episode_timesteps=}, {episode_return=}')
            print('=================================================================')

        episode_timesteps+=1
        episode_returns.append(episode_return)
        episode_costs.append(episode_cost)

    assert(not self.drem)
    if plot:
        xplot = np.arange(len(episode_returns))
        plt.figure(figsize=(len(episode_returns)*0.2,8))
        plt.bar(xplot, episode_returns)
        plt.xlabel('episode')
        plt.ylabel('return')
        plt.grid(axis='both')
        plt.xticks(xplot, fontsize=8, rotation=90, labels=xplot+1)
        #plt.show()

        plt.figure(figsize=(len(episode_costs)*0.2,8))
        plt.bar(xplot, episode_costs, color='tab:green')
        plt.xlabel('episode')
        plt.ylabel('cost')
        plt.grid(axis='both')
        plt.xticks(xplot, fontsize=8, rotation=90, labels=xplot+1)
        #plt.show()

    mean_episode_return = np.mean(episode_returns)
    sum_episode_cost = np.sum(episode_costs)

    total_episodes = len(episode_returns)
    print(f'Avg Episode Return: {mean_episode_return} over [{total_episodes}] episodes')
    print(f'Total Episode Cost: {sum_episode_cost} over [{total_episodes}] episodes')
    #print(f'cost : {total_cost} over [{total_episodes}] episodes')
    
    return np.array(self.history), mean_episode_return, sum_episode_cost, total_episodes
