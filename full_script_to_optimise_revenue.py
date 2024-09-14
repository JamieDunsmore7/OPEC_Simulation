import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit


### STAGE 1: GET A RELATIONSHIP FOR OIL PRICE VS OPEC PRODUCTION ###

# Get the historical data from the table
# If someone has these on an excel sheet then I should be able to read in from there too
historic_prices = np.array([83.2, 87.2, 91.5, 94.1, 95.4, 96.6, 98.9, 100.6, 102.6, 106.6, 108.8])
historic_prices_per_thousand_barrels = historic_prices * 1000
historic_world_production = np.array([87096, 86716, 86306, 86062, 85941, 85819, 85606, 85439, 85256, 84876, 84664])
historic_OPEC_production = np.array([43000, 40500, 37800, 36200, 35400, 34600, 33200, 32100, 30900, 28400, 27000])
historic_ROW_production = historic_world_production - historic_OPEC_production


# Both the demand and ROW supply curve are linear functions in the historic data. If this changes, will need a new function.
def linear_function(x, slope, intercept):
    return slope*x + intercept

demand_slope, demand_intercept = curve_fit(linear_function, historic_world_production, historic_prices_per_thousand_barrels)[0]
print('Demand')
print(demand_slope, demand_intercept)

row_supply_slope, row_supply_intercept = curve_fit(linear_function, historic_ROW_production, historic_prices_per_thousand_barrels)[0]
print('ROW Supply')
print(row_supply_slope, row_supply_intercept)


# Plot the data and the linear functions
plt.scatter(historic_world_production, historic_prices_per_thousand_barrels)
plt.plot(historic_world_production, linear_function(np.array(historic_world_production), demand_slope, demand_intercept), label='Demand Curve')
plt.scatter(historic_ROW_production, historic_prices_per_thousand_barrels)
plt.plot(historic_ROW_production, linear_function(np.array(historic_ROW_production), row_supply_slope, row_supply_intercept), label='ROW Supply Curve')
plt.ylabel('Price ($ per thousand barrels)')
plt.xlabel('Capacity (thousands of barrels)')
plt.legend()
plt.tight_layout()
plt.show()



# get the x value from the y value
def rearrange_linear_function(slope, intercept, y):
    return (y - intercept) / slope




def calculate_oil_price(OPEC_production):
    '''
    INPUT: OPEC Production for a given round (thousands of barrels)
    OUTPUT: Optimal price per 1000 barrels to maximize OPEC revenue
    NOTE: assumes that the demand and ROW supply curves are linear functions
    '''
    def objective(price):
        total_demand = rearrange_linear_function(demand_slope, demand_intercept, price) # for a given price, calculate the total demand
        row_capacity = total_demand - OPEC_production
        row_price = linear_function(row_capacity, row_supply_slope, row_supply_intercept) # for a ROW capacity, calculate the corresponding price
        # In market equilibrium, the price at which consumers are willing to buy should equal the price at which ROW producers are willing to sell
        return abs(row_price - price)  # Minimize the absolute difference between supply and demand

    # Find the price that gives market equilibrium
    result = minimize(objective, x0=0, bounds=[(-1000000, 1000000)])
    optimal_price = result.x[0]
    return optimal_price


# As a sanity check, make sure that the calculate oil price function reproduces the historic data
list_of_prices = []
list_of_OPEC_production = np.arange(0, 100000, 1000)
for OPEC_production in list_of_OPEC_production:
    optimal_price = calculate_oil_price(OPEC_production)
    list_of_prices.append(optimal_price)

list_of_prices = np.array(list_of_prices) # convert to numpy array for plotting

plt.title('Estimated Oil Price as a function of OPEC Production')
plt.plot(list_of_OPEC_production, list_of_prices/1000, label='Model')
plt.scatter(historic_OPEC_production, historic_prices, label = 'Historic Data')
plt.ylabel('Price ($ per barrel)')
plt.xlabel('OPEC Production (thousands of barrels)')
plt.legend()
plt.show()


# Relationship between OPEC production and price is LINEAR: get the coefficients
optimum_params = curve_fit(linear_function, historic_OPEC_production, historic_prices)[0]
print(optimum_params) # THESE ARE THE COEFFICIENTS!









### STAGE 2: RUN THE OPTIMISATION ###

# Define the total number of rounds and other constants
no_of_rounds = 10
current_round = 1
interest_rate = 1.05
final_value_per_barrel = 80
initial_reserves = 262400 + 115000 + 137620 + 99377 + 104000 + 97800 + 75000

# Price per barrel as a function of OPEC production
def price_per_barrel(production):
    return optimum_params[0] * production + optimum_params[1] # Price per barrel decreases with increasing OPEC production


# Objective function to maximize
def revenue_function(x, R): # x is the vector of oil produced in each round
    '''
    x: vector containing oil production in each round
    R: total reserves at the start of the game
    '''
    total_revenue = 0
    for i in range(no_of_rounds + 1 - current_round):
        oil_produced = x[i]
        revenue = oil_produced * price_per_barrel(oil_produced)
        no_of_turns_left = (no_of_rounds + 1) - i # since there is technically an '11th' round where we can generate interest
        revenue_including_interest = revenue * (interest_rate ** (no_of_turns_left))
        total_revenue += revenue_including_interest

    # Add the value of remaining reserves at the end of the game
    remaining_reserves = R - np.sum(x)
    total_revenue += final_value_per_barrel * remaining_reserves

    return -total_revenue  # Minimise the negative of revenue to maximise revenue

# CONSTRAINT: Total oil produced should not exceed the reserves
def constraint(x, R):
    return R - np.sum(x)

constraints = {'type': 'ineq', 'fun': constraint, 'args': (initial_reserves,)} # Define the constraint in a form compatible with scipy.optimize

# CONSTRAINT: Production must be positive in every round
bounds = [(0, None) for _ in range(no_of_rounds)]

# Initial guess of oil produced in each round for the optimisation
initial_guess = np.full(no_of_rounds+1-current_round, initial_reserves / no_of_rounds)

# Perform the optimization
result = minimize(revenue_function, initial_guess, args=(initial_reserves,), constraints=constraints, bounds=bounds)

# Output the optimal production for each round
optimal_production = result.x
total_revenue = -result.fun

print("Optimal oil production in each round:", optimal_production)
print("Total maximum revenue:", total_revenue)







##### STAGE 3: PLOT THE RESULTS #####

# Plot 1: Optimum OPEC Oil Production
xvalues = np.arange(current_round, no_of_rounds + 1)

plt.figure(figsize=(10, 6))
plt.scatter(xvalues, optimal_production, marker='x', color='blue', s=100)
plt.plot(xvalues, optimal_production, color='blue', linestyle='--')
plt.title('Optimal OPEC Oil Production per Round', fontsize=16, pad=20)
plt.xlabel('Round', fontsize=14)
plt.ylabel('Optimum OPEC Oil Production (thousands of barrels)', fontsize=14)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('optimal_oil_production.png')
plt.show()

# Plot 2: Target Oil Price over Time
prices = [price_per_barrel(production) for production in optimal_production]
plt.figure(figsize=(10, 6))
plt.scatter(xvalues, prices, marker='x', color='green', s=100)
plt.plot(xvalues, prices, color='green', linestyle='--')
plt.title('Target Oil Price per Round', fontsize=16, pad=20)
plt.xlabel('Round', fontsize=14)
plt.ylabel('Target Oil Price ($ per barrel)', fontsize=14)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('target_oil_price.png')
plt.show()





