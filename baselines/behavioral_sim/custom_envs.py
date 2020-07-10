import gym
from gym import spaces
import numpy as np
from behavioral_sim import utils
from behavioral_sim.agents import *
from behavioral_sim.reward import Reward


class BehavSimEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space_string="continuous", one_day=False, energy_in_state=False, 
    response ='t', yesterday_in_state = False, day_of_week=False):
        super(BehavSimEnv, self).__init__()
        self.action_length = 10
        self.action_subspace = 3
        self._create_action_space(action_space_string)
        self.yesterday_in_state = yesterday_in_state
        self.prev_ideal = []
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        if(yesterday_in_state):
            if(energy_in_state):
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

        else:
            if energy_in_state:
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.response = response
        self.one_day = one_day
        self.energy_in_state = energy_in_state
        self.prices = self._get_prices(one_day)
        assert self.prices.shape == (365, 10)

        self.player_dict = self._create_agents()
        self.cur_iter = 0
        self.day = 0
        self.day_of_week_flag = day_of_week
        self.day_of_week = self.days_of_week[self.day % 5]
        self.prev_energy = np.array([80, 120, 200, 210, 180, 250, 380, 310, 220, 140])
        print("BehavSimEnv Initialized")

    def _create_action_space(self, action_space_string):
        action_space_string = action_space_string.lower()
        self.action_space_string = action_space_string

        if action_space_string == "continuous":
            self.action_space = spaces.Box(low=0, high=10, shape=(self.action_length,), dtype=np.float32)
        elif action_space_string == "symmetric":
            self.action_space = spaces.Box(low=-10, high=10, shape=(self.action_length,), dtype=np.float32)
        elif action_space_string == "multidiscrete":
            discrete_space = [self.action_subspace] * self.action_length
            self.action_space = spaces.MultiDiscrete(discrete_space)
        elif action_space_string == "discrete":
            self.action_space = spaces.Discrete(self.action_subspace ** self.action_length)
        else:
            print("action_space not recognized. Defaulting to MultiDiscrete.")
            discrete_space = [self.action_subspace] * self.action_length
            self.action_space = spaces.MultiDiscrete(discrete_space)
            self.action_space_string = "multidiscrete"



    def _get_prices(self, one_day):
        
        all_prices = []
        if one_day:
            # if repeating the same day, then use a random day. 
            # SET FIXED DAY HERE
            day = 184
            price = utils.price_signal(day + 1)
            price = np.array(price[8:18])
            price = np.maximum([0.01], price)
            naive_reward = Reward([], price, 35, 300)
            for i in range(365):
                all_prices.append(price)
        else:
            day = 0
            for i in range(365):  
                price = utils.price_signal(day + 1)
                price = np.array(price[8:18])
                # put a floor on the prices so we don't have negative prices
                # price = np.maximum([0.01], price)
                all_prices.append(price)
                day += 1

        return np.array(all_prices)

    def _create_agents(self):
        """Initialize the market agents
            Args:
              None

            Return:
              agent_dict: dictionary of the agents
        """

        player_dict = {}

        # I dont trust the data at all
        # helper comment         [0, 1, 2, 3, 4, 5,  6,  7,  8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18, 19, 20,  21, 22, 23]
        sample_energy = np.array([0, 0, 0, 0, 0, 0, 20, 50, 80, 120, 200, 210, 180, 250, 380, 310, 220, 140, 100, 50, 20,  10,  0,  0])

        #only grab working hours (8am - 6pm)
        working_hour_energy = sample_energy[8:18]

        my_baseline_energy = pd.DataFrame(data={"net_energy_use": working_hour_energy})

        player_dict['player_0'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 10, response= 't')
        player_dict['player_1'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 10, response='t')
        player_dict['player_2'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 10, response ='s')
        player_dict['player_3'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 10,response ='s')
        player_dict['player_4'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 10,response ='s')
        player_dict['player_5'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 10,response ='l')
        player_dict['player_6'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 10, response ='l')
        player_dict['player_7'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 10, response ='l')

        if(self.response != 'm'):
            for player in player_dict.values():
                player.response = self.response
        # player_dict['player_0'] = MananPerson1(my_baseline_energy, points_multiplier = 10)
        # player_dict['player_1'] = MananPerson1(my_baseline_energy, points_multiplier = 10)
        # player_dict['player_2'] = MananPerson1(my_baseline_energy, points_multiplier = 10)
        # player_dict['player_3'] = MananPerson1(my_baseline_energy, points_multiplier = 10)
        # player_dict['player_4'] = MananPerson1(my_baseline_energy, points_multiplier = 10)
        # player_dict['player_5'] = MananPerson1(my_baseline_energy, points_multiplier = 10)
        # player_dict['player_6'] = MananPerson1(my_baseline_energy, points_multiplier = 10)
        # player_dict['player_7'] = MananPerson1(my_baseline_energy, points_multiplier = 10)

        return player_dict

    def step(self, action):
        prev_observation = self.prices[self.day]
        self.day = (self.day + 1) % 365
        self.day_of_week = self.days_of_week[self.day % 5]
        self.cur_iter += 1
        next_observation = self.prices[self.day]
        if self.cur_iter > 0:
            done = True
        else:
            done = False
        points = self._points_from_action(action)
        energy_consumptions = self._simulate_humans(prev_observation, points)
        if(self.yesterday_in_state):
            observation = np.concatenate((prev_observation, self.prev_ideal))
        if self.energy_in_state:
            # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same
            self.prev_energy = energy_consumptions["avg"]
            observation = np.concatenate((self.prev_energy, next_observation))
        else:
            observation = next_observation

        reward = self._get_reward(prev_observation, energy_consumptions)
        info = {}
        return observation, reward, done, info

    def _points_from_action(self, action):
        if self.action_space_string == "discrete":
            points = [0] * self.action_length
            temp = action
            for i in range(self.action_length-1, -1, -1):
                points[i] = temp // (self.action_subspace**i)
                temp = temp % (self.action_subspace**i)
        else:
            points = action
        return points

    def _simulate_humans(self, prev_observation, action):
        energy_consumptions = {}
        total_consumption = np.zeros(10)
        num_players = 0
        for player_name in self.player_dict:
            if player_name != "avg":
                player = self.player_dict[player_name]
                # get the points output from players
                # CHANGE PLAYER RESPONSE FN HERE
                # player_energy = np.array(player.threshold_exp_response(action))

                # player_energy = player.predicted_energy_behavior(action, self.day % 5)
                if(self.day_of_week_flag):
                    player_energy = player.get_response(action,day_of_week=self.day_of_week)
                else:
                    player_energy = player.get_response(action,day_of_week=None)

                energy_consumptions[player_name] = player_energy
                total_consumption += player_energy
                num_players += 1
        energy_consumptions["avg"] = total_consumption / num_players
        return energy_consumptions


    def _get_reward(self, prev_observation, energy_consumptions):
        total_reward = 0

        for player_name in energy_consumptions:
            if player_name != "avg":
                # get the points output from players
                player = self.player_dict[player_name]

                # get the reward from the player's output
                player_min_demand = player.get_min_demand()
                player_max_demand = player.get_max_demand()
                player_energy = energy_consumptions[player_name]
                player_reward = Reward(player_energy, prev_observation, player_min_demand, player_max_demand)
                player_ideal_demands = player_reward.ideal_use_calculation()
                #self.prev_ideal = player_ideal_demands
                # either distance from ideal or cost distance
                # distance = player_reward.neg_distance_from_ideal(player_ideal_demands)
                reward = player_reward.cost_distance(player_ideal_demands)

                total_reward += reward
        return total_reward

    def get_reward_planning_model(self, prev_observation, energy_consumptions):
        """
        This function is meant to generate a reward when we have one agent handling the whole office 
        so, the planning model produces one energy consumption estimate per the whole office, I guess
        and we then paste that onto each person.

        prev_observation: grid prices from yesterday, 10-dim vector
        energy_consumption: energy consumption predicted from the planning model

        In the future, this function can be modified to only calculate the reward from a single person, instead of looping over the whole set. 

        """


        total_reward = 0

        ###### potential hack alert: changing a dictionary based on players' energy to a dictionary based on their names
        
        player_names = self.player_dict.keys()
        for player_name in player_names:
            if player_name != "avg":
                # get the points output from players
                player = self.player_dict[player_name]

                # get the reward from the player's output
                player_min_demand = player.get_min_demand()
                player_max_demand = player.get_max_demand()
                player_energy = energy_consumptions
                player_reward = Reward(player_energy, prev_observation, player_min_demand, player_max_demand)
                player_ideal_demands = player_reward.ideal_use_calculation()
                #self.prev_ideal = player_ideal_demands
                # either distance from ideal or cost distance
                # distance = player_reward.neg_distance_from_ideal(player_ideal_demands)
                reward = player_reward.cost_distance(player_ideal_demands)

                total_reward += reward
        return total_reward
  
    def reset(self):
        if self.energy_in_state:
            return np.concatenate((self.prices[self.day], self.prev_energy))
        else:
            return self.prices[self.day]

    def render(self, mode='human'):
        pass

    def close (self):
        pass


class HourlySimEnv(BehavSimEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space_string="continuous", one_day=False, energy_in_state=False, 
    response = 't', yesterday_in_state = False, day_of_week=False):
        self.action_length = 1
        self.action_subspace = 3
        self._create_action_space(action_space_string)
        self.yesterday_in_state = yesterday_in_state
        self.prev_ideal = []
        if(yesterday_in_state):
            if(energy_in_state):
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        else:
            if energy_in_state:
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.response = response
        self.one_day = one_day
        self.energy_in_state = energy_in_state
        self.prices = self._get_prices(one_day)
        assert self.prices.shape == (365, 10)
        
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        self.player_dict = self._create_agents()
        self.cur_iter = 0
        self.day = 0
        self.hour = 0
        self.day_of_week_flag = day_of_week
        self.day_of_week = self.days_of_week[self.day % 5]
        #TODO sample from wg1.txt
        self.prev_energy = [80, 120, 200, 210, 180, 250, 380, 310, 220, 140]
        self.prev_points = [0]
        print("HourlySimEnv Initialized")

    def step(self, action):
        
        #self.day = (self.day + 1) % 365
        self.hour += 1
        self.cur_iter += 1
        point = self._points_from_action(action)
        second_half = np.array([self.hour],dtype=np.float32)
        self.prev_points.append(point)
        prev_observation = self.prices[self.day]
        if self.hour == 10:
            #assert len(self.prev_points) == 11
            points = np.squeeze(np.array(self.prev_points[1:]))
            energy_consumptions = self._simulate_humans(prev_observation, points)
            # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same
            self.prev_energy = energy_consumptions["avg"]
            reward = self._get_reward(prev_observation, energy_consumptions)
            self.day = (self.day + 1) % 365
            self.day_of_week = self.days_of_week[self.day % 5]
            self.reset()
            done = True
        else:
            reward = np.array(0)
            done = False
        
        if self.yesterday_in_state:
            observation = np.concatenate((prev_observation, self.prev_ideal))
            if(self.energy_in_state):
                observation = np.concatenate((observation, self.prev_energy, second_half))
            else:
                observation = np.concatenate((observation, second_half))
        else:
            if self.energy_in_state:
                observation = np.concatenate((self.prices[self.day], self.prev_energy, second_half))
            else:
                observation = np.concatenate((self.prices[self.day], second_half))

        
        info = {}
        print(observation.shape)
        return observation, reward, done, info
  
    def reset(self):
        self.hour = 0
        self.prev_points = [0]
        if self.energy_in_state:
            observation = np.concatenate((self.prices[self.day], self.prev_energy, np.array([0]), np.array([self.hour])))
        else:
            observation = np.concatenate((self.prices[self.day], np.array([0]), np.array([self.hour])))
        return observation

