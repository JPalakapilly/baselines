import torch
import torch.nn as nn
import csv
import numpy as np
from scipy.optimize import minimize
import os


def price_signal(day = 45):

    """
    Utkarsha's work on price signal from a building with demand and solar

    Input: Day = an int signifying a 24 hour period. 365 total, all of 2012, start at 1.
    Output: netdemand_price, a measure of how expensive energy is at each time in the day
        optionally, we can return the optimized demand, which is the building
        calculating where the net demand should be allocated
    """

    pv = np.array([])
    price = np.array([])
    demand = np.array([])
    data_path = os.path.join(os.getcwd(), "building_data.csv") #Took out "baselines", "behavioral_sim",
    with open(data_path, encoding='utf8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader,None)
        rowcount = 0
        for row in csvreader:
            pv = np.append(pv, 0.001*float(row[3])) # Converting Wh to kWh
            price = np.append(price, float(row[2])) # Cost per kWh
            val = row[5]
            if val in (None,""): #How to treat missing values
                val = 0
            else:
                val = float(val) # kWh
            demand = np.append(demand, val)
            rowcount+=1
            # if rowcount>100:
            #     break

    pvsize = 5 #Assumption

    netdemand = demand.copy()
    for i in range(len(demand)):
        netdemand[i] = demand[i] - pvsize*pv[i]

    # Data starts at 5 am on Jan 1
    netdemand_24 = netdemand[24*day-5:24*day+19]
    price_24 = price[24*day-5:24*day+19]
    pv_24 = pv[24*day-5:24*day+19]
    demand_24 = demand[24*day-5:24*day+19]

    # Calculate optimal load scheduling. 90% of load is fixed, 10% is controllable.
    def optimise_24h(netdemand_24, price_24):
        currentcost = netdemand_24*price_24

        fixed_load = 0.9*netdemand_24
        controllable_load = sum(0.1*netdemand_24)
        # fixed_load = 0*netdemand_24
        # controllable_load = sum(netdemand_24)

        def objective(x):
            load = fixed_load + x
            cost = np.multiply(price_24,load)
            # Negative demand means zero cost, not negative cost
            # Adding L1 regularisation to penalise shifting of occupant demand
            lambd = 0.005
            return sum(np.maximum(cost,0)) + lambd*sum(abs(x-0.1*netdemand_24))

        def constraint_sumofx(x):
            return sum(x) - controllable_load

        def constraint_x_positive(x):
            return x

        x0 = np.zeros(24)
        cons = [
            {'type':'eq', 'fun': constraint_sumofx},
            {'type':'ineq', 'fun':constraint_x_positive}
        ]
        sol = minimize(objective, x0, constraints=cons)
        return sol

    sol = optimise_24h(netdemand_24,price_24)
    x = sol['x']

    netdemand_price_24 = netdemand_24*price_24

    return(netdemand_price_24)
    
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
def copy_params(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)