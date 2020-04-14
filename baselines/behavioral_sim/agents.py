import pandas as pd
import numpy as np
import cvxpy as cvx

#### file to make the simulation of people that we can work with 


class Person():
	""" Person (parent?) class -- will define how the person takes in a points signal and puts out an energy signal 
	baseline_energy = a list or dataframe of values. This is data from SinBerBEST 
	points_multiplier = an int which describes how sensitive each person is to points 

	"""

	def __init__(self, baseline_energy_df, points_multiplier = 1):
		self.baseline_energy_df = baseline_energy_df
		self.baseline_energy = np.array(self.baseline_energy_df["net_energy_use"])
		self.points_multiplier = points_multiplier
		
		baseline_min = self.baseline_energy.min()
		baseline_max = self.baseline_energy.max()
		baseline_range = baseline_max - baseline_min
		
		self.min_demand = np.maximum(0, baseline_min + baseline_range * .05)
		self.max_demand = np.maximum(0, baseline_min + baseline_range * .95)


	def energy_output_simple_linear(self, points):
		"""Determines the energy output of the person, based on the formula:
		
		y[n] = -sum_{rolling window of 5} points + baseline_energy + noise

		inputs: points - list or dataframe of points values. Assumes that the 
		list will be in the same time increment that energy_output will be. 

		For now, that's in 1 hour increments

		"""
		points_df = pd.DataFrame(points)
		
		points_effect = (
			points_df
				.rolling(
						window = 5,
						min_periods = 1)
				.mean()
			)



		time = points_effect.shape[0]
		energy_output= []

		for t in range(time):
			temp_energy = self.baseline_energy[t] - points_effect.iloc[t]*self.points_multiplier + \
				np.random.normal(1)
			energy_output.append(temp_energy)
			
		return pd.DataFrame(energy_output)

	def pure_linear_signal(self, points, baseline_day=0):
		"""
		A linear person. The more points you give them, the less energy they will use
		(within some bounds) for each hour. No rolling effects or anything. The simplest
		signal. 
		"""

		# hack here to always grab the first day from the baseline_energy
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]

		points_effect = np.array(points * self.points_multiplier)
		output = output - points_effect

		# impose bounds/constraints
		output = np.maximum(output, self.min_demand)
		output = np.minimum(output, self.max_demand)
		return output



	def get_min_demand(self):
		return self.min_demand
		# return np.quantile(self.baseline_energy, .05)

	def get_max_demand(self):
		return self.max_demand
		# return np.quantile(self.baseline_energy, .95)

class Person_with_hysteresis(Person):
	""" Wendy -- Determines the energy output of the person, based on the formula:
		
		y[n] = f(points) + baseline_energy + noise

		f: super special secret function that Wendy designs with hysteresis 

		inputs: points - list or dataframe of points values. Assumes that the 
		list will be in the same time increment that energy_output will be. 

		For now, that's in 5 minute increments"""

	def __init__(self, baseline_energy, points_multiplier = 1):
		pass

class FixedDemandPerson(Person):

	def __init__(self, baseline_energy_df, points_multiplier = 1):
		super().__init__(baseline_energy_df, points_multiplier)


	def demand_from_points(self, points, baseline_day=0):
		# hack here to always grab the first day from the baseline_energy
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]
		total_demand = np.sum(output)


		points_effect = np.array(points * self.points_multiplier)
		output = output - points_effect

		# scale to keep total_demand (almost) constant
		# almost bc imposing bounds afterwards
		output = output * (total_demand/np.sum(output))

		# impose bounds/constraints
		output = np.maximum(output, self.min_demand)
		output = np.minimum(output, self.max_demand)

		return output

	def adverserial_linear(self, points, baseline_day=0):
		# hack here to always grab the first day from the baseline_energy
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]
		total_demand = np.sum(output)


		points_effect = np.array(points * self.points_multiplier)
		output = output + points_effect

		# scale to keep total_demand (almost) constant
		# almost bc imposing bounds afterwards
		output = output * (total_demand/np.sum(output))

		# impose bounds/constraints
		output = np.maximum(output, self.min_demand)
		output = np.minimum(output, self.max_demand)

		return output

class DeterministicFunctionPerson(Person):

	def __init__(self, baseline_energy_df, points_multiplier = 1):
		super().__init__(baseline_energy_df, points_multiplier)

	def threshold_response_func(self, points):
		points = np.array(points) * self.points_multiplier
		threshold = np.mean(points)
		return [p if p>threshold else 0 for p in points]

	def exponential_response_func(self, points):
		points = np.array(points) * self.points_multiplier
		points_effect = [p**2 for p in points]

		return points_effect

	def sin_response_func(self,points):
		points = np.array(points) 
		# n = np.max(points)
		# points = [np.sin((float(i)/float(n))*np.pi) for i in points]	
		points = [np.sin(float(i)*np.pi)*self.points_multiplier for i in points]	
		points = points 
		return points

	def routine_output_transform(self, points_effect, baseline_day=0):
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]
		total_demand = np.sum(output)

		# scale to keep total_demand (almost) constant
		# almost bc imposing bounds afterwards
		output = output - points_effect
		output = output * (total_demand/np.sum(output))

		# impose bounds/constraints
		output = np.maximum(output, self.min_demand)
		output = np.minimum(output, self.max_demand)
		return output

	def threshold_response(self, points):
		points_effect = self.threshold_response_func(points)
		output = self.routine_output_transform(points_effect)
		return output

	def sin_response(self, points):
		points_effect = self.sin_response_func(points)
		output = self.routine_output_transform(points_effect)
		return output

	def exp_response(self, points):
		points_effect = self.exponential_response_func(points)
		output = self.routine_output_transform(points_effect)
		return output

	def threshold_exp_response(self,points):
		points_effect = self.exponential_response_func(points)
		points_effect = self.threshold_response_func(points_effect)
		output = self.routine_output_transform(points_effect)
		return output

	def linear_response(self, points):
		points_effect = points*self.points_multiplier
		output = self.routine_output_transform(points_effect)
		return output


class MananPerson1(Person):

	def __init__(self, baseline_energy_df, points_multiplier=.8):
		# ignores baseline_energy_df
		# this is just for backwards compatability

		self.baseline_energy_hour = 300
		self.day_of_week_multiplier = np.array([1.1, 1.15, 1, 0.9, 0.8])
		self.hour_multiplier = np.array([0.8, 0.9, 1, 0.9, 0, 0.9, 1.1, 1.1, 1.0, 0.9])
		self.AFFINITY_TO_POINTS = points_multiplier
		self.ENERGY_STD_DEV = 5

		self.baseline_energy_day = np.array(self.baseline_energy_hour * self.hour_multiplier)
		self.total_baseline_day = np.sum(self.baseline_energy_day)*self.day_of_week_multiplier
		
		self.min_demand = self.baseline_energy_day.min()*self.day_of_week_multiplier.min()
		self.max_demand = self.baseline_energy_day.max()*self.day_of_week_multiplier.max()
		

		self.MAX_DIFFERENTIAL = 20
	
	def redistributed_energy(self, points, day_num):


		energy_curve = cvx.Variable(len(points))
		objective = cvx.Minimize(energy_curve.T * points)
		constraints = [
			cvx.sum(energy_curve, axis=0, keepdims=True)
			== self.total_baseline_day[day_num]
		]
		for hour in range(10):
			constraints += [energy_curve[hour] >= 0]

		for hour in range(1, 10):
			constraints += [
					cvx.abs(energy_curve[hour] - energy_curve[hour - 1])
					<= self.MAX_DIFFERENTIAL
			]

		problem = cvx.Problem(objective, constraints)
		problem.solve()
		return energy_curve.value

	def predicted_energy_behavior(self, points, day_num):

		perfect_energy_use = self.redistributed_energy(points, day_num)
		baseline_energy_use = self.baseline_energy_day*self.day_of_week_multiplier[day_num]

		means = np.empty(len(perfect_energy_use))
		for i in range(len(perfect_energy_use)):
			lesser, greater = (
				(perfect_energy_use[i], baseline_energy_use[i])
				if perfect_energy_use[i] < baseline_energy_use[i]
				else (baseline_energy_use[i], perfect_energy_use[i])
			)
			means[i] = lesser + 0.8 * (greater - lesser)
		sample = np.random.normal(means, self.ENERGY_STD_DEV)
		return np.maximum(np.zeros(sample.shape), sample)