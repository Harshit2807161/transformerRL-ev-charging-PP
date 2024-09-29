import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.spaces
from math import ceil
from .common import REMAP, pjs
def log_evaluations(evaluations_path):
    E = np.load(evaluations_path)
    # E.files # ['timesteps', 'results', 'ep_lengths']
    ts, res, epl = E['timesteps'], E['results'], E['ep_lengths']
    # ts.shape, res.shape, epl.shape #<---- (eval-freq, n_eval_episodes)
    resM, eplM = np.mean(res, axis=1), np.mean(epl, axis=1) # mean reward of all eval_episodes

    plt.figure(figsize=(8,4))
    plt.scatter(ts, resM, color='green')
    plt.plot(ts, resM, label='mean_val_reward', color='green')
    plt.xlabel('step')
    plt.ylabel('mean_val_reward')
    plt.legend()
    #plt.show()
    plt.clf()
    plt.close()


    plt.figure(figsize=(8,4))
    plt.scatter(ts, eplM, color='blue')
    plt.plot(ts, eplM, label='mean_episode_len', color='blue')
    plt.xlabel('step')
    plt.ylabel('mean_episode_len')
    plt.legend()
    #plt.show()
    plt.clf()
    plt.close()

    E.close()
    return

class ENV(gym.Env):
    
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    # know constants - NOTE: ALWAYS USE CAPSLOCK FOR KNOWN CONSTANTS
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    E_MAX = 24 # kWh
    E_MIN = 2.4 #kWh
    
    CE =  100
    MK =  0.015
    TAU = 0.01
    
    X=7
    P0=0.01
    P1=4
    P2=4
    P3=2
    P4=0.5
    P5=0.5
    X1=0.9
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    def __init__(self, 
                price_dataset,    # object of class PriceData
                price_predictor,  # a callable like f(ps.reshape(1,seqlen,1)) :-> feature_vector (1-dimensional)
                average_past=0,
                random_days=False,  
                frozen=False,
                discrete=True,
                seed=None, 
                record=0 # 0=no record, 1=record_episode, 2=record_all
                ) -> None:
        super().__init__()
        self.price_dataset = price_dataset
        self.average_past=average_past
        self.n_days = self.price_dataset.n_days
        self.signal = self.price_dataset.signal
        self.length = self.price_dataset.length
        self.signal_max = np.max(self.signal)
        self.signal_min = np.min(self.signal)
        
        
        self.price_predictor = price_predictor  
        self.oracle = (self.price_predictor.model is None)
        
        self.pS_past, self.pS_future = self.price_predictor.past, self.price_predictor.future
        #self.pS_delta = pS_past + pS_future
        self.start_at = max(2,ceil(self.pS_past/24)) 
        self.end_at = self.n_days - max(3,(ceil(self.pS_future/24)+1))
        assert(self.end_at>self.start_at)
        self.day_count = len(range(self.start_at, self.end_at))
        self.random_days=random_days
        
        self.rng = np.random.default_rng(seed)
        self.record = record
        self.drem=[]
        self.discrete = discrete
        self.build_spaces()
        
        self.frozen=frozen
        self.started=False


    def build_spaces(self):
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        # build state space
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        self.pS_len = self.pS_past + self.pS_future
        self.pp_dim = (  self.pS_len if self.oracle else self.pS_future )
        self.observation_dim = 1 + 1 + (1 if self.average_past>0 else 0) + self.pp_dim
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, dtype= np.float32, shape=(self.observation_dim,))
        self.observation = np.zeros(shape=self.observation_space.shape, dtype=self.observation_space.dtype)
        self.obs_soc =  self.observation[0:1]
        self.obs_ts =  self.observation[1:2]

        if self.average_past>0:
            self.obs_mp = self.observation[2:3] #  =  mean of past price (self.average_past) no of steps
            self.obs_feature = self.observation[3:]
        else:
            self.obs_feature = self.observation[2:]


        self.obs_prices = (np.zeros(self.pS_len-1, dtype=self.observation_space.dtype))
        self.obs_past =  self.obs_prices[:self.pS_past]
        self.obs_future = self.obs_prices[self.pS_past:]

        #self.state_dim = sum(self.state_dim_list)
        #self.state_space = gym.spaces.Box(low = -np.inf, high = np.inf, dtype= np.float32, shape=(self.state_dim,))
        #self.state = np.zeros(shape=self.state_space.shape, dtype=self.state_space.dtype)
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        # build action space
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        if self.discrete:
            self.action_space = gym.spaces.Discrete(7)
            self.action_mapper = lambda a: [6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0][a]
            #self.action_mapper = REMAP(Input_Range=(-1,1), Mapped_Range=(-6, 6))
        else:
            self.action_space = gym.spaces.Box(low = -6.0, high = 6.0, dtype= np.float32, shape=(1,))
            self.action_mapper = lambda a: a[0]
        

    def start(self):
        emark = [] # episode markers
        tmark = []
        socmark = []
        #print("#######################################")
        for day in range(self.start_at, self.end_at):
            # note day i starts at index i*24
            # for each day generate a arrival, departure time
            arrival_time = int((np.clip(self.rng.normal(loc=18.0, scale=1), 15, 21)).astype('int'))
            departure_time = int((np.clip(self.rng.normal(loc=8.0, scale=1), 6, 11)).astype('int'))
            start_index = day*24 + arrival_time
            end_index = (day+1)*24 + departure_time
            tmark.append( (  (day,arrival_time), (day+1,departure_time) ))
            emark.append( (start_index, end_index) )
            socmark.append( float( (np.clip(self.rng.normal(loc=0.5, scale=0.1), 0.2, 0.8))*self.E_MAX ))

        assert(self.day_count == len(emark))
        self.emark = emark #np.array(emark)
        self.tmark = tmark #np.array(tmark)
        self.socmark = socmark
        self.started=True
        
    def restart(self):
        if self.record: self.history = []
        self.episode = 0
        # randomly assign 
        # skip start_at*24 hours
        if self.frozen:
            if not self.started: self.start()
        else: self.start()
        
        if self.random_days:
            self.drem = list ( np.random.choice(np.arange(self.day_count), self.day_count, replace=False) )
        else:
            self.drem = list(   np.arange(self.day_count)[::-1]   )
        
    def get_features(self):
        #print("i is: ",self.i)
        self.obs_prices = self.signal[self.i-self.pS_past : self.i+self.pS_future-1]
        #print(f"Initial: {self.obs_prices}")
        if self.average_past>0: self.obs_mp[:] = np.mean(self.signal[self.i-self.average_past :self.i])
        self.obs_feature = self.obs_prices
        self.obs_feature = np.append(self.obs_feature,self.price_predictor(self.i-49))
        #print(f"Latter: {self.obs_feature}")

    def reset(self):
        if len(self.drem) == 0 : self.restart() # all episodes ended , do a restart
        ei = self.drem.pop()
        self.obs_soc[:] = self.socmark[ei]
        self.ts=0
        self.start_index, self.end_index = self.emark[ei]
        (self.start_day, self.arrival_time), (self.end_day, self.departure_time) = self.tmark[ei]
        self.ts2go = (24 - self.arrival_time) + self.departure_time
        
        self.obs_ts[:]= self.ts2go # self.ts
        
        self.i = self.start_index + self.ts
        self.current_price = self.signal[self.i]
        self.get_features()
        self.episode+=1
        return self.observation

    def step(self, action):
        act = self.action_mapper(action)
        soc_before = self.obs_soc[0]
        self.obs_soc[:] = self.obs_soc + act # update battery
        
        
        '''
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # reward calculation model view paper
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        common_reward = (-self.current_price*act - (self.CE*self.MK*0.01*(act/self.E_MAX))) # common part of reward function
        if self.i>= self.end_index:
            reward =common_reward-0.01*(((self.E_MAX-self.obs_soc[0])**2)  )
            done=True
        else:
            reward=common_reward
            done=False
        '''
        
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # reward calculation-2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        common_reward = -self.X*self.current_price*act # common part of reward function
        if  self.i>=self.end_index:
            reward =-self.P3*((self.E_MAX-self.obs_soc[0])**2)
            done=True
        else: 
            if self.obs_soc[0]>self.E_MAX: 
                reward=-self.P1*((self.E_MAX-self.obs_soc[0])**2)
            else:
                if self.obs_soc[0]<self.E_MIN:
                    reward=self.P2*act*((self.E_MIN-self.obs_soc[0])**2)
                else:
                    reward=common_reward  
            done=False
        '''
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # reward calculation-3 Final: current work 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        common_reward = -self.current_price*self.P0*act # common part of reward function
        if  self.i>=self.end_index:
            reward =common_reward-self.P4*((self.E_MAX-self.obs_soc[0]))
            done=True
        else: 
            if self.obs_soc[0]>self.E_MAX: 
                reward=common_reward-self.P4*((self.obs_soc[0]-self.E_MAX))
            else:
                if self.obs_soc[0]<self.E_MIN:
                    reward=common_reward-self.P4*((self.E_MIN-self.obs_soc[0]))
                else:
                    # NOTE: <--- self.obs_mp[0] does not exist if self.average_past is less than or equal to 0
                    reward=common_reward  + self.P5*act*self.P0*(self.obs_mp[0]-self.current_price) 
            done=False
        
            
        self.obs_soc[:] = np.clip( self.obs_soc, 0, self.E_MAX) # clip update battery
        soc_after = self.obs_soc[0]
        soc_delta = soc_after - soc_before
        step_cost =  (soc_delta)*self.current_price

        if self.record: self.history.append(  (
            self.i, 
            self.obs_soc[0], 
            self.current_price, 
            (0 if self.ts else -self.episode), 
            act,
            soc_delta,
            step_cost)  )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.i+=1
        self.ts+=1 # update MDP time-step
        self.ts2go-=1
        self.obs_ts[:]= self.ts2go #self.ts
        self.current_price = self.signal[self.i]
        
        self.get_features()

        if done and self.record: self.history.append(  (self.i, self.obs_soc[0], self.current_price, self.episode, 0, 0, 0)   ) 
        return self.observation, float(reward), bool(done), \
            {
                'step_cost':step_cost,
                'soc_delta':soc_delta,
                'pxa': self.current_price*act,
                'soc_after': soc_after,
            }

    def render(self, mode=0):
        # to render: obs
        ts2go = int(self.obs_ts[0])
        t = self.ts
        Et = self.obs_soc[0]
        Pt = self.current_price
        _Pt = self.obs_past
        Pt_ = self.obs_future
        Ft = self.obs_feature
        Mt = (self.obs_mp[0] if self.average_past>0 else None)
        index = self.i
        if mode==0:
            print(f"""\t[{t=}] :: {index=}:: {ts2go=} {Et= } {Mt=} : {Pt= } """)
        elif mode==1:
            print(f"""\t[{t=}] :: {index=}:: {ts2go=} {Et= } :{Mt=}  {Pt= } : {Ft= }""")
        else:
            print(f"""
            [#{t=}] 
            {Et=}
            {Mt=}
            {_Pt=}
            {Pt=}
            {Pt_=}
            {Ft=}
            """)

    def test(self, model, pp_model_name, verbose=1, render=True, rmode=0, plot=True, result_dir=None):
        # When performing this test, 
        # restart the environemnt and make sure random_days is false
        
        self.restart() # <--restart once
        max_episodes = self.day_count
        print(f'Testing Environment: [{self.price_dataset.csv}] : for [{max_episodes=}]')

        if not self.frozen: print('WARNING:: recomended using [frozen = True]')
        if self.random_days: print('WARNING:: recomended using [random_days = False]')
        
        print(f'{self.frozen=}, {self.random_days=}\n')
        #Pt = self.current_price
        #-------------------
        episode = 0
        episode_returns = []
        episode_costs= []
        episode_C = []
        while episode<max_episodes:
            
            episode+=1
            episode_return = 0
            episode_cost=0
            episode_timesteps, done = 0, False

            episode_PA_sum = 0

            obs = self.reset()
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
                episode_PA_sum += info['pxa']
                e_dep = info['soc_after']
                episode_cost+=step_cost
                
                if verbose>0: 
                    print(f'--> [t={episode_timesteps}] :: [{act=}:->{self.action_mapper(act)}] : {rew=:.2f} ~{done=} ~{step_cost= }')
                if render: self.render(mode=rmode)



            if verbose>0:
                print(f'\nDone: {episode_timesteps=}, {episode_return=}, {episode_cost=}')
                print('=================================================================')

            episode_timesteps+=1
            first_price = self.signal[self.start_index + episode_timesteps]

            episode_returns.append(episode_return)
            episode_costs.append(episode_cost)

            episode_C.append(first_price * (self.E_MAX - e_dep) + episode_cost)

        assert(not self.drem)
        xplot = np.arange(len(episode_returns))
        #added bellow
        #xplot = np.arange(start=1, stop=len(episode_returns), step=10)
        #plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        if plot:
            
            fig = plt.figure(figsize=(len(episode_returns)*0.2,8))
            plt.bar(xplot, episode_returns, color='tab:blue')
            plt.xlabel('episode')
            plt.ylabel('return')
            plt.grid(axis='both')
            plt.xticks(xplot, fontsize=8, rotation=90, labels=xplot+1)
            #plt.show()
            fig.savefig(pjs(result_dir, f'EP_return_{pp_model_name}.png'))
            plt.clf()
            plt.close()

            fig = plt.figure(figsize=(len(episode_costs)*0.2,8))
            plt.bar(xplot, episode_costs, color='tab:green')
            plt.xlabel('episode')
            plt.ylabel('cost')
            plt.grid(axis='both')
            plt.xticks(xplot, fontsize=8, rotation=90, labels=xplot+1)
            #plt.show()
            fig.savefig(pjs(result_dir, f'EC_return_{pp_model_name}.png'))
            plt.clf()
            plt.close()

            fig = plt.figure(figsize=(10, 6))
            plt.plot(xplot, episode_C, color='tab:purple')
            plt.xlabel('episode')
            plt.ylabel('cummulative cost')
            plt.grid(axis='both')
            plt.xticks(xplot, fontsize=8, rotation=90, labels=xplot+1)
            #plt.show()
            fig.savefig(pjs(result_dir, f'CC_return_{pp_model_name}.png'))
            plt.clf()
            plt.close()
            # Open a text file in write mode
            with open(f"./{result_dir}/output_{pp_model_name}.txt", "w") as file:
                # Loop through the array and write each value to the file
                for i in range(len(episode_returns)):
                    file.write(str(episode_returns[i]) + "\n")

            # Close the file
            file.close()

        mean_episode_return = np.mean(episode_returns)
        sum_episode_cost = np.sum(episode_costs)
        # added bellow
        sum_episode_C=np.sum(episode_C)
        sum_episode_returns = np.sum(episode_return) 
        

        total_episodes = len(episode_returns)
        print(f'Avg Episode Return: {mean_episode_return} over [{total_episodes}] episodes')
        print(f'Total Episode Cost: {sum_episode_cost} over [{total_episodes}] episodes')
        print(f'Total Cummulative Episode Return: {sum_episode_returns} over [{total_episodes}] episodes')
        # added bellow
        print(f'Total cumulative_Episode Cost: {sum_episode_C} over [{total_episodes}] episodes')
        #print(f'cost : {total_cost} over [{total_episodes}] episodes')

        
        return np.array(self.history), mean_episode_return, sum_episode_cost, total_episodes, xplot, np.array(episode_returns), np.array(episode_costs)


    def segment_plot(self, duration_per_graph, history=None, fix_ylim=0, 
                     max_segments=0, legend=False, episode_index=True, grid=False, price_scale=24.0, cost_scale=24.0):
        #duration_per_graph = 75 #<------------ steps per graph
        if history is None: history = np.array(self.history)

        d, l, s, il = int(duration_per_graph), len(self.signal), 0, []
        while True:
            e = s+d
            if e>=l:
                il.append((s, l))
                break
            else:
                il.append((s, e))
                s=e
        print(f'signal-length: {l},\nsegments:[#{len(il)}]\n{il}')  
        if max_segments>0:
            segments = il[:max_segments]
        elif max_segments<0:
            segments = il[max_segments:]
        else:
            segments=il
        
        figures=[]
        for (sig_from, sig_to) in segments:
            gx = history[:,0]
            gix=np.where((gx>=sig_from)&(gx<sig_to))[0]
            gx = history[gix, 0] 

            gsoc = history[gix,1]
            gpt = history[gix,2]*(price_scale)
            gst = history[gix,3]
            gact = history[gix,4]
            #gdelta = history[gix, 5]
            gcost = history[gix, 6]*(cost_scale)
            #gep = history[gix,5]

            ss = self.signal[sig_from:sig_to]
            print(f'{sig_from=}, {sig_to=}, {len(ss)=}')
            fig = plt.figure(figsize=(len(ss)*0.15,4), constrained_layout=True)
            figures.append( (sig_from, sig_to, fig) )
            plt.xlabel('time')
            plt.xlim((sig_from-1,sig_to))
            if fix_ylim: plt.ylim((-15, fix_ylim))
            plt.xticks(np.arange(sig_from-1,sig_to), fontsize=8, rotation=90)
            if grid: plt.grid(axis='both')
            
            #plt.plot(gx, gep, color='red')

            plt.bar(np.arange(len(ss))+sig_from, ss, 0.1, color='black')
            

            plt.bar(gx, gsoc, color='orange', label='SoC')
            plt.bar(gx, gact, 0.5, color='blue', label='action')
            plt.scatter(gx, gact, color='blue', marker='.')

            #plt.bar(gx-0.3, gdelta, 0.25, color='pink',  label='soc_delta')
            plt.bar(gx+0.3, gcost, 0.25,  color='brown', label='step_cost')

            csa = np.where(gst<0)[0]
            xcsa, ycsa = gx[csa], gst[csa]
            plt.scatter(xcsa, ycsa*0-10, color='green', label='episode_start')
            plt.bar(xcsa, ycsa*0-10, 0.1, color='green')
            if episode_index:
                for s,x in zip(ycsa, xcsa): plt.annotate(f'#[{int(-s)}]', xy = (x, -8), fontsize=8)

            csz = np.where(gst>0)[0]
            xcsz, ycsz = gx[csz], gst[csz]
            plt.scatter(xcsz, ycsz*0-10, color='red', label='episode_end')
            plt.bar(xcsz, ycsz*0-10, 0.1, color='red')
            
            plt.scatter(gx, gpt, color='black', label='price', marker='.')
            
            #plt.bar(range(len(ss)), ss*price_scale, 0.25, color='black')
            
            plt.bar(gx, gpt, 0.1, color='black') 
            
            
            plt.plot(np.arange(len(ss))+sig_from,ss*price_scale, color='black', linewidth=0.5)
            #for s,x in zip(ycsz, xcsz):
            #    plt.annotate(f'{s}]', xy = (x, -11))

            if legend: plt.legend()
            #plt.show()
            plt.close()
        return figures


"""
~-~-~-~-~-~-~-~-~-~-
ARCHIVE 
~-~-~-~-~-~-~-~-~-~-
"""
