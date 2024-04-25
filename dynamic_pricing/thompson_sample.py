import pandas as pd
import numpy as np
from scipy import stats
import pymc as pm

import matplotlib.pyplot as plt


def plot_demand_curve(ax, price_point, price_demand_dict, demand_prior_mean, max_curve=25):
    '''
    Shared function. Plot the demand curve based on the prior parameters.

    Parameters:
    - ax (Axes): The Matplotlib axes object to plot on.
    - price_point (array-like): The array of price points.
    - price_demand_dict (dict): A dictionary containing price-demand pairs.
    - demand_prior_mean (float): The mean demand based on the prior parameters.
    - max_curve (int, optional): The maximum number of curves to plot.

    Returns:
    - None
    '''

    for i in price_demand_dict.keys():
        ax.plot(price_point, price_demand_dict[i], color='silver')
        if i == max_curve:
            break

    ax.plot(price_point, demand_prior_mean, color='red')
    ax.set_ylim(0, 10)
    ax.set_title('Initial Prior of the Demand Curve')


def plot_profit_curve(ax, price_point, price_profit_dict, profit_prior_mean, max_curve=25):
    '''
    Shared function. Plot the profit curve based on the prior parameters.

    Parameters:
    - ax (Axes): The Matplotlib axes object to plot on.
    - price_point (array-like): The array of price points.
    - price_profit_dict (dict): A dictionary containing price-profit pairs.
    - profit_prior_mean (float): The mean profit based on the prior parameters.
    - max_curve (int, optional): The maximum number of curves to plot.

    Returns:
    - None
    '''

    for i in price_profit_dict.keys():
        ax.plot(price_point, price_profit_dict[i], color='silver')
        # set a limit to make the plot readable
        # we have 4,000 data points, so let's limit it at 25
        if i == max_curve:
            break

    ax.plot(price_point, profit_prior_mean, color='red')

    ax.axline(xy1=(0, 0), slope=0, color='black', linestyle='--')
    ax.set_title('Initial Prior of the Profit Curve')


def create_df_parameters(a, eta, sd):
    '''
    Shared function. Create a DataFrame containing updated prior parameters.

    Parameters:
    - a (array-like): The 'a' parameter values.
    - eta (array-like): The price elasticity parameter values.
    - sd (array-like): The standard deviation values.

    Returns:
    - df_posterior (DataFrame): The DataFrame containing posterior parameters.
    '''

    df_parameters = pd.DataFrame(columns=['a', 'eta', 'sd'])

    df_parameters['a'] = a
    df_parameters['eta'] = eta
    df_parameters['sd'] = sd

    return df_parameters


def create_price_demand_dict(price_point, df_parameters):
    '''
    Shared function. Create a dictionary of price-demand pairs based on updated prior parameters.

    Parameters:
    - price_point (float): The price point for which to calculate the demand.
    - df_posterior (DataFrame): The DataFrame containing posterior parameters.

    Returns:
    - price_demand_dict (dict): A dictionary containing price-demand pairs.
    '''

    price_demand_dict = {}

    for i in range(len(df_parameters)):
        demand = df_parameters.loc[i, 'a'] * price_point**(df_parameters.loc[i, 'eta'])
        price_demand_dict[i] = demand

    return price_demand_dict


def create_price_profit_dict(price_point, fixed_cost, cogs, price_demand_dict, df_parameters):
    '''
    Shared function. Create a dictionary of price-profit pairs based on posterior parameters.

    Parameters:
    - price_point (float): The price point for which to calculate the profit.
    - fixed_cost (float): The fixed cost associated with the product.
    - cogs (float): The cost of goods sold.
    - price_demand_dict (dict): A dictionary containing price-demand pairs.
    - df_posterior (DataFrame): The DataFrame containing posterior parameters.

    Returns:
    - price_profit_dict (dict): A dictionary containing price-profit pairs.
    '''

    price_profit_dict = {}

    for i in range(len(df_parameters)):
        price_profit_dict[i] = price_point * price_demand_dict[i] - (cogs * price_demand_dict[i] + fixed_cost)

    return price_profit_dict


class PriorIntitialization:
    '''
    Class for prior initialization to analyze dynamic pricing modeling using Thompson Sampling.
    '''

    def __init__(self, data, price_col, qty_col, fixed_cost=None, var_cost=None, price_point=None):
        '''
        Initialize the PriorInitialization class.

        Parameters:
        - data (DataFrame): The DataFrame containing the data.
        - price_col (str): The name of the column containing price data.
        - qty_col (str): The name of the column containing quantity data.
        - fixed_cost (float, optional): The fixed cost associated with the product.
        - var_cost (float, optional): The variable cost associated with the product.
        '''

        self.price = data[price_col].values
        self.demand = data[qty_col].values
        self.fixed_cost = fixed_cost
        self.var_cost = var_cost
        self.price_point = price_point

    
    def initialize_prior(self):
        '''
        Initialize the prior distributions by updating the constant-elasticity demand function with current data that we have.
        '''

        price = self.price
        demand = self.demand

        with pm.Model() as model:
            a = pm.Normal('a', mu=0, sigma=2)
            eta = pm.Normal('eta', mu=0.5, sigma=0.5)
            sd = pm.Exponential('sd', lam=1)

            # price elasticity function, as the mean for our demand function 
            def price_elasticity(p, a=a, eta=eta):
                mu = a * p ** -eta
                return mu

            # update posterior based on the demand data we have
            # pass price to the price_elasticity, and demand observed to the D
            D = pm.Normal('D', mu=price_elasticity(price), sigma=sd, observed=demand)
            trace_prior = pm.sample(1000)

        self.trace_prior = trace_prior

        self.a_prior = trace_prior.posterior['a'].values.flatten()
        self.eta_prior = trace_prior.posterior['eta'].values.flatten()
        self.sd_prior = trace_prior.posterior['sd'].values.flatten()

        self.a_prior_mean = self.a_prior.mean()
        self.eta_prior_mean = self.eta_prior.mean()
        self.sd_prior_mean = self.sd_prior.mean()

    
    def investigate_prior_parameters(self):
        '''
        Visualize and analyze the prior parameters for a, price elasticiy, and standard deviation.
        '''

        pm.plot_trace(self.trace_prior)

        plt.tight_layout()
        plt.show()

        print('updated a value from the result:', round(self.a_prior_mean, 2))
        print('updated price elasticity (eta) value from the result:', round(self.eta_prior_mean, 2))
        print('updated standard deviation from the result:', round(self.sd_prior_mean, 2))

    
    def params_for_curve(self):
        '''
        Calculate the parameters for plotting demand and profit curves.

        Returns:
        - None
            The method stores the calculated parameters in the object's attributes for later use.
        '''
        df_parameters = create_df_parameters(self.a_prior, self.eta_prior, self.sd_prior)
        price_demand_dict = create_price_demand_dict(self.price_point, df_parameters)
        price_profit_dict = create_price_profit_dict(self.price_point, self.fixed_cost, self.var_cost, price_demand_dict, df_parameters)

        demand_mean = self.a_prior_mean * self.price_point**(self.eta_prior_mean)
        profit_mean = self.price_point * demand_mean - (self.var_cost * demand_mean + self.fixed_cost)

        self.df_parameters = df_parameters
        self.price_demand_dict = price_demand_dict
        self.price_profit_dict = price_profit_dict

        self.demand_mean = demand_mean
        self.profit_mean = profit_mean
    

    def plot_curve(self, max_curve=25, curve_type='profit'):
        '''
        Plot the curve based on the prior parameters.

        Parameters:
        - ax (Axes): The Matplotlib axes object to plot on.
        - price_point (array-like): The array of price points.
        - max_curve (int, optional): The maximum number of curves to plot.
        - curve_type (str, optional): The type of curve to plot ('profit' or 'demand').

        Returns:
        - None: The plot is displayed using Matplotlib's plt.show() function.
        '''

        fig, ax = plt.subplots(1, 1)

        if curve_type == 'profit':
            plot_profit_curve(ax, self.price_point, self.price_profit_dict, self.profit_mean, max_curve=max_curve)
            return plt.show()

        elif curve_type == 'demand':
            plot_demand_curve(ax, self.price_point, self.price_demand_dict, self.demand_mean, max_curve=max_curve)
            return plt.show()

        else:
            raise ValueError('curve_type should be either profit or demand.')
    
    
    # property to save all the properties of the class
    # will be used for the next class to do the incremental update and thompson sampling

    @property
    def a_prior_value(self):
        return self.a_prior

    @property
    def eta_prior_value(self):
        return self.eta_prior
    
    @property
    def sd_prior_value(self):
        return self.sd_prior
    
    @property
    def demand_mean_value(self):
        return self.demand_mean
    
    @property
    def profit_mean_value(self):
        return self.profit_mean
    
    @property
    def df_parameters_value(self):
        return self.df_parameters
    
    @property
    def price_demand_dict_value(self):
        return self.price_demand_dict
    
    @property
    def price_profit_dict_value(self):
        return self.price_profit_dict


class ThompsonSampling:
    '''
    Thompson Sampling algorithm to decide at which price should we go next, whether to explore new reward or to exploit known reward.
    '''

    def __init__(self, a_prior, eta_prior, sd_prior, price_to_test, var_cost, fixed_cost):
        '''
        Initialize the ThompsonSampling class.

        Parameters:
        - a_prior (array-like): Array of prior 'a' values for each price level.
        - eta_prior (array-like): Array of prior 'eta' values for each price level.
        - sd_prior (array-like): Array of prior standard deviation values for each price level.
        - price_to_test (array-like): Array of prices to test.
        - var_cost (float): Variable cost per unit.
        - fixed_cost (float): Fixed cost.
        '''

        self.var_cost = var_cost
        self.a_prior = a_prior
        self.eta_prior = eta_prior
        self.sd_prior = sd_prior
        self.fixed_cost = fixed_cost
        self.price_to_test = price_to_test

    
    def sampling(self):
        '''
        Perform Thompson Sampling to sample 'a', 'eta', and 'sd' values.
        '''
        index = np.random.randint(0, len(self.a_prior))

        # extract the corresponding values for 'a', 'eta', and 'sd'
        self.a_thomp_sample = self.a_prior[index]
        self.eta_thomp_sample = self.eta_prior[index]
        self.sd_thomp_sample = self.sd_prior[index]

        print("sampled values: a =", self.a_thomp_sample, ", eta =", self.eta_thomp_sample, ", sd =", self.sd_thomp_sample)


    def calculate_rewards(self):
        '''
        Calculate rewards for each price level based on Thompson Sampling values.
       
        Returns:
        - DataFrame: DataFrame containing prices, demand, and profit for each price level.
        '''
        thomp_sample_df = pd.DataFrame(columns=['price', 'demand', 'profit'])

        for i, price in enumerate(self.price_to_test):
            demand_thomp_sample = self.a_thomp_sample * price**(self.eta_thomp_sample)
            profit_thomp_sample = price * demand_thomp_sample - (self.var_cost * demand_thomp_sample + self.fixed_cost)

            thomp_sample_df.loc[i, 'price'] = price
            thomp_sample_df.loc[i, 'demand'] = demand_thomp_sample
            thomp_sample_df.loc[i, 'profit'] = profit_thomp_sample

        thomp_sample_df = thomp_sample_df.sort_values('profit')

        return thomp_sample_df
    

class IncrementalUpdatePrior:
    '''
    Class to perform incremental update of prior parameters for a dynamic pricing model.
    '''

    def __init__(self, prior_params, fixed_cost, var_cost, data_observed, price_col, qty_col, price_point):
        '''
        Initialize the IncrementalUpdatePrior class.

        Parameters:
        - prior_params (dict): Dictionary containing prior parameters for 'a', 'eta', 'sd', 'price_demand_prior', 'price_profit_prior', 'demand_mean_prior', and 'profit_mean_prior'.
        - fixed_cost (float): Fixed cost.
        - var_cost (float): Variable cost per unit.
        - data_observed (DataFrame): DataFrame containing observed data.
        - price_col (str): Column name for prices in the observed data.
        - qty_col (str): Column name for quantities in the observed data.
        - price_point (float): Price point for analysis.
        '''
        
        self.a_prior = prior_params['a_prior']
        self.eta_prior = prior_params['eta_prior']
        self.sd_prior = prior_params['sd_prior']

        self.price_demand_prior = prior_params['price_demand_prior']
        self.price_profit_prior = prior_params['price_profit_prior']
        
        self.demand_mean_prior = prior_params['demand_mean_prior']
        self.profit_mean_prior = prior_params['profit_mean_prior']

        self.fixed_cost = fixed_cost 
        self.var_cost = var_cost

        self.price = data_observed[price_col].values
        self.qty = data_observed[qty_col].values

        self.price_point = price_point
    

    @staticmethod
    def from_posterior(param, samples, k=100):
        '''
        Create an interpolated distribution from posterior samples.
        Used for doing the incremental update on samples we got from pyMC object.

        Parameters:
        - param (str): Parameter name ('a', 'eta', or 'sd').
        - samples (array-like): Posterior samples.
        - k (int): Number of points for interpolation.

        Returns:
        - pm.Interpolated: Interpolated distribution for the parameter.
        '''
        smin, smax = np.min(samples), np.max(samples)
        width = smax - smin
        x = np.linspace(smin, smax, k)
        y = stats.gaussian_kde(samples)(x)
        
        # what was never sampled should have a small probability but not 0,
        # so we'll extend the domain and use linear approximation of density on it
        x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
        y = np.concatenate([[0], y, [0]])
        return pm.Interpolated(param, x, y)


    def update_prior(self):
        '''
        Perform an incremental update of prior parameters using observed data.
        '''

        with pm.Model() as m1:
            # from the function above, we can update our a, eta, and sd parameters based on the updated posterior
            a = self.from_posterior('a', self.a_prior)
            eta = self.from_posterior('eta', self.eta_prior)
            sd = self.from_posterior('sd', self.sd_prior)

            def price_elasticity(p, a=a, eta=eta):
                mu = a * p ** -eta
                return mu

            # now, we can pass the data we sampled before to the observed parameter
            # to do the incremental update
            D = pm.Normal('D', mu=price_elasticity(self.price), sigma=sd, observed=self.qty)
            
            trace_posterior = pm.sample(5000, tune=5000)

        self.a_posterior = trace_posterior.posterior['a'].values.flatten()
        self.eta_posterior = trace_posterior.posterior['eta'].values.flatten()
        self.sd_posterior = trace_posterior.posterior['sd'].values.flatten()

        self.a_posterior_mean = self.a_posterior.mean()
        self.eta_posterior_mean = self.eta_posterior.mean()
        self.sd_posterior_mean = self.sd_posterior.mean()

        self.trace_posterior = trace_posterior


    def investigate_posterior_parameters(self):
        '''
        Visualize and analyze the prior parameters for a, price elasticiy, and standard deviation.
        '''

        pm.plot_trace(self.trace_posterior)

        plt.tight_layout()
        plt.show()

        print('posterior of a value from the result:', round(self.a_posterior_mean, 2))
        print('posterior of price elasticity (eta) value from the result:', round(self.eta_posterior_mean, 2))
        print('posterior standard deviation from the result:', round(self.sd_posterior_mean, 2))
        
    
    def params_for_curve(self):
        '''
        Calculate the parameters for plotting demand and profit curves.

        Returns:
        - None
            The method stores the calculated parameters in the object's attributes for later use.
        '''
        df_parameters = create_df_parameters(self.a_posterior, self.eta_posterior, self.sd_posterior)
        price_demand_dict = create_price_demand_dict(self.price_point, df_parameters)
        price_profit_dict = create_price_profit_dict(self.price_point, self.fixed_cost, self.var_cost, price_demand_dict, df_parameters)

        demand_mean = self.a_posterior_mean * self.price_point**(self.eta_posterior_mean)
        profit_mean = self.price_point * demand_mean - (self.var_cost * demand_mean + self.fixed_cost)

        self.df_parameters_posterior = df_parameters
        self.price_demand_posterior = price_demand_dict
        self.price_profit_posterior = price_profit_dict

        self.demand_mean_posterior = demand_mean
        self.profit_mean_posterior = profit_mean
        
    
    def plot_prior_posterior(self, max_curve=25, curve_type='profit'):
        '''
        Plot prior and posterior curves for demand or profit.

        Parameters:
        - max_curve (int): Maximum value for the curve.
        - curve_type (str): Type of curve ('profit' or 'demand').
        '''

        if curve_type == 'profit':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

            plot_profit_curve(ax1, self.price_point, self.price_profit_prior, self.profit_mean_prior, max_curve=max_curve)
            plot_profit_curve(ax2, self.price_point, self.price_profit_posterior, self.profit_mean_posterior, max_curve=max_curve)

            ax1.set_title('Price to Profit - Previous Beliefs about Prior')
            ax2.set_title('Price to Profit - After Updating Prior (Posterior)')
            
            return plt.show()

        elif curve_type == 'demand':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            
            plot_demand_curve(ax1, self.price_point, self.price_demand_prior, self.demand_mean_prior)
            plot_demand_curve(ax2, self.price_point, self.price_demand_posterior, self.demand_mean_posterior)

            ax1.set_title('Price to Demand - Previous Beliefs about Prior')
            ax2.set_title('Price to Demand - After Updating Prior (Posterior)')
            
            return plt.show()

        else:
            raise ValueError('curve_type should be either profit or demand.')


    # property to save all the properties of the class
    # will be used for the next class to do the incremental update and thompson sampling

    @property
    def a_posterior_value(self):
        return self.a_posterior

    @property
    def eta_posterior_value(self):
        return self.eta_posterior
    
    @property
    def sd_posterior_value(self):
        return self.sd_posterior
    
    @property
    def demand_mean_posterior_value(self):
        return self.demand_mean_posterior
    
    @property
    def profit_mean_posterior_value(self):
        return self.profit_mean_posterior
    
    @property
    def df_parameters_value(self):
        return self.df_parameters_posterior
    
    @property
    def price_demand_dict_value(self):
        return self.price_demand_posterior
    
    @property
    def price_profit_dict_value(self):
        return self.price_profit_posterior
