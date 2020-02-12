import gym
from gym import spaces
import numpy as np
from baselines.behavioral_sim import utils
from baselines.behavioral_sim.agents import *
from baselines.behavioral_sim.reward import Reward


class BehavSimEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, one_day=False, energy_in_state=False):
        super(BehavSimEnv, self).__init__()

        discrete_space = [3] * 10
        self.action_space = spaces.MultiDiscrete(discrete_space)
        if energy_in_state:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.one_day = one_day
        self.energy_in_state = energy_in_state
        self.prices = self._get_prices(one_day)
        assert self.prices.shape == (365, 10)

        self.player_dict = self._create_agents()
        self.cur_iter = 0
        self.day = 0
        print("BehavSimEnv Initialized")


    def _get_prices(self, one_day):
        
        all_prices = []
        if one_day:
            # if repeating the same day, then use a random day. 
            # SET FIXED DAY HERE
            day = np.random.randint(365)
            price = utils.price_signal(day + 1)
            price = np.array(price[8:18])
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

        # TODO: This needs to be updated

        # Skipping rows b/c data is converted to PST, which is 16hours behind
        # so first 10 hours are actually 7/29 instead of 7/30
        
        # baseline_energy1 = convert_times(pd.read_csv("wg1.txt", sep = "\t", skiprows=range(1, 41)))
        # baseline_energy2 = convert_times(pd.read_csv("wg2.txt", sep = "\t", skiprows=range(1, 41)))
        # baseline_energy3 = convert_times(pd.read_csv("wg3.txt", sep = "\t", skiprows=range(1, 41)))

        # be1 = change_wg_to_diff(baseline_energy1)
        # be2 = change_wg_to_diff(baseline_energy2)
        # be3 = change_wg_to_diff(baseline_energy3)

        player_dict = {}

        # I dont trust the data at all
        # helper comment         [0, 1, 2, 3, 4, 5,  6,  7,  8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18, 19, 20,  21, 22, 23]
        sample_energy = np.array([0, 0, 0, 0, 0, 0, 20, 50, 80, 120, 200, 210, 180, 250, 380, 310, 220, 140, 100, 50, 20,  10,  0,  0])

        #only grab working hours (8am - 6pm)
        working_hour_energy = sample_energy[8:18]

        my_baseline_energy = pd.DataFrame(data={"net_energy_use": working_hour_energy})

        player_dict['player_0'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
        player_dict['player_1'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
        player_dict['player_2'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
        player_dict['player_3'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
        player_dict['player_4'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
        player_dict['player_5'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
        player_dict['player_6'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
        player_dict['player_7'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)

        return player_dict

    def step(self, action):
        prev_observation = self.prices[self.day]
        self.day = (self.day + 1) % 365
        self.cur_iter += 1
        observation = self.prices[self.day]
        if self.cur_iter > 0:
            done = True
        else:
            done = False
        energy_consumptions = self._simulate_humans(prev_observation, action)
        if self.energy_in_state:
            observation = np.concatenate(observation, energy_consumptions["avg"])
        reward = self._get_reward(energy_consumptions)
        info = {}
        return observation, reward, done, info

    def _simulate_humans(self, prev_observation, action):
        energy_consumptions = {}
        total_consumption = np.zeros(10)
        num_players = 0
        for player_name in self.player_dict:

            player = self.player_dict[player_name]
            # get the points output from players
            # CHANGE PLAYER RESPONSE FN HERE
            player_energy = player.threshold_exp_response(action)
            energy_consumptions[player_name] = player_energy
            total_consumption += player_energy
            num_players += 1
        energy_consumptions["avg"] = total_consumption/num_players
        return energy_consumptions


    def _get_reward(self, energy_consumptions):
        total_reward = 0
        for player_name in energy_consumptions:

            # get the points output from players
            player = self.player_dict[player_name]

            # get the reward from the player's output
            player_min_demand = player.get_min_demand()
            player_max_demand = player.get_max_demand()
            player_reward = Reward(player_energy, prev_observation, player_min_demand, player_max_demand)
            player_ideal_demands = player_reward.ideal_use_calculation()
            # either distance from ideal or cost distance
            # distance = player_reward.neg_distance_from_ideal(player_ideal_demands)
            reward = player_reward.scaled_cost_distance(player_ideal_demands)

            total_reward += reward
        return total_reward
  
    def reset(self):
        self.day = np.random.randint(365)
        self.cur_iter = 0
        return self.prices[self.cur_iter]

    def render(self, mode='human'):
        pass

    def close (self):
        pass