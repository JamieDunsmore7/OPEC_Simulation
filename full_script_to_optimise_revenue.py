import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit


### STAGE 1: GET A RELATIONSHIP FOR OIL PRICE VS OPEC PRODUCTION ###

# Get the historical data from the table
historic_prices = np.array([83.2, 87.2, 91.5, 94.1, 95.4, 96.6, 98.9, 100.6, 102.6, 106.6, 108.8])
historic_prices_per_thousand_barrels = historic_prices * 1000
historic_world_production = np.array([87096, 86716, 86306, 86062, 85941, 85819, 85606, 85439, 85256, 84876, 84664])
historic_OPEC_production = np.array([43000, 40500, 37800, 36200, 35400, 34600, 33200, 32100, 30900, 28400, 27000])
historic_ROW_production = historic_world_production - historic_OPEC_production


# Both the demand and ROW supply curve are linear functions in the historic data. If this changes, will need a new function.
def linear_function(x, slope, intercept):
    return slope*x + intercept

# Get coefficients for the demand function and ROW supply function (both are linear)
demand_slope, demand_intercept = curve_fit(linear_function, historic_world_production, historic_prices_per_thousand_barrels)[0]
row_supply_slope, row_supply_intercept = curve_fit(linear_function, historic_ROW_production, historic_prices_per_thousand_barrels)[0]


# useful function when converting between prices and quanitties
def rearrange_linear_function(slope, intercept, y):
    return (y - intercept) / slope



# Function to calculate the oil price from the OPEC production, given the fact that total supply = demand.
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


# SANITY CHECK: Does the function correctly predict historic relationship between price and OPEC production?
list_of_prices = []
list_of_OPEC_production = np.arange(0, 100000, 1000)
for OPEC_production in list_of_OPEC_production:
    optimal_price = calculate_oil_price(OPEC_production) # our prediction for oil price given OPEC production
    list_of_prices.append(optimal_price)

list_of_prices = np.array(list_of_prices) # convert to numpy array for plotting

# Plot the historic oil price as a function of OPEC production
plt.title('Estimated Oil Price as a function of OPEC Production')
plt.plot(list_of_OPEC_production, list_of_prices/1000, label='Model')
plt.scatter(historic_OPEC_production, historic_prices, label = 'Historic Data')
plt.ylabel('Price ($ per barrel)')
plt.xlabel('OPEC Production (thousands of barrels)')
plt.legend()
plt.show()

# Relationship between OPEC production and price is LINEAR: get the coefficients
optimum_params = curve_fit(linear_function, historic_OPEC_production, historic_prices)[0]
print('COEFFICIENTS OF THE OIL PRICE VS OPEC PRODUCTION FUNCTION')
print(optimum_params) # THESE ARE THE COEFFICIENTS!


### STAGE 2: RUN THE OPTIMISATION ###

# CALCULATE OPTIMIMUM STRATEGY FROM ROUND 9
no_of_rounds = 10
current_round = 9
interest_rate = 1.05
final_value_per_barrel = 80
weighted_extraction_cost_per_barrel = 6.7 # weighted averaged of extraction costs
initial_reserves = 262400 + 115000 + 137620 + 99377 + 104000 + 97800 + 75000 - 53000 - 40400 - 38356 - 53100 - 42084 - 41828 - 48554 - 39100

# Price per barrel in dollars as a function of OPEC production in thousands of barrels
def price_per_barrel(production):
    return optimum_params[0] * production + optimum_params[1] # Price per barrel decreases with increasing OPEC production



# Calculate the profit given initial reserves and the oil production in each round.
def profit_function(x, R): # x is the vector of oil produced in each round
    '''
    x: vector containing oil production in each round
    R: total reserves at the start of the game
    '''
    total_profit = 0
    for i in range(no_of_rounds + 1 - current_round):
        oil_produced = x[i]
        revenue = oil_produced * price_per_barrel(oil_produced)
        profit = revenue - (oil_produced * weighted_extraction_cost_per_barrel)
        no_of_turns_left = (no_of_rounds + 1 - current_round) - i # since there is technically an '11th' round where we can generate interest
        profit_including_interest = profit * (interest_rate ** (no_of_turns_left))
        total_profit += profit_including_interest

    # Add the value of remaining reserves at the end of the game
    remaining_reserves = R - np.sum(x)
    total_profit += (final_value_per_barrel - weighted_extraction_cost_per_barrel) * remaining_reserves

    return -total_profit  # Minimise the negative of revenue to maximise revenue

# CONSTRAINT: Total oil produced should not exceed the reserves
def constraint(x, R):
    return R - np.sum(x)

constraints = {'type': 'ineq', 'fun': constraint, 'args': (initial_reserves,)} # Define the constraint in a form compatible with scipy.optimize

# CONSTRAINT: Production must be positive in every round
bounds = [(0, None) for _ in range(no_of_rounds + 1 - current_round)]

# initial guess of production in each round to pass into the optimisation
initial_guess = np.full(no_of_rounds+1-current_round, initial_reserves / no_of_rounds)

# Run the optimisation
result_profit = minimize(profit_function, initial_guess, args=(initial_reserves,), constraints=constraints, bounds=bounds)


optimal_production_profit = result_profit.x
total_profit = -result_profit.fun

print('OPTIMUM SOLUTION FROM ROUND 9 ONWARDS')
print("Optimal oil production in each round:", optimal_production_profit)
print("Total maximum profit:", total_profit)


#### STAGE 3: RUN THE OPTIMISATION WITH THE OPTIMAL STRATEGY FROM ROUND 1 ####

# Define the total number of rounds and other constants
no_of_rounds = 10
current_round = 1
interest_rate = 1.05
final_value_per_barrel = 80
weighted_extraction_cost_per_barrel = 6.7
initial_reserves = 262400 + 115000 + 137620 + 99377 + 104000 + 97800 + 75000

constraints = {'type': 'ineq', 'fun': constraint, 'args': (initial_reserves,)} # Define the constraint in a form compatible with scipy.optimize

# CONSTRAINT: Production must be positive in every round
bounds = [(0, None) for _ in range(no_of_rounds + 1 - current_round)]

# Initial guess of oil produced in each round for the optimisation
initial_guess = np.full(no_of_rounds+1-current_round, initial_reserves / no_of_rounds)

result_profit_old = minimize(profit_function, initial_guess, args=(initial_reserves,), constraints=constraints, bounds=bounds)
optimal_production_profit_old = result_profit_old.x
total_profit_old = -result_profit_old.fun

print("Optimal oil production in each round:", optimal_production_profit_old)
print("Total maximum profit:", total_profit_old)








##### STAGE 4: PLOT THE RESULTS #####

# Plot 1: Optimum OPEC Oil Production
xvalues = np.arange(1, no_of_rounds + 1)
xvalues_old = np.arange(1, no_of_rounds + 1)


plt.figure(figsize=(8, 6))

plt.scatter(xvalues, optimal_production_profit_old, marker='x', color='red', s=100, label='Profit Maximising')
plt.plot(xvalues, optimal_production_profit_old, color='red', linestyle='--')

plt.title('Optimal OPEC Oil Production per Round', fontsize=16, pad=20)
plt.xlabel('Round', fontsize=14)
plt.ylabel('Optimum OPEC Oil Production (thousands of barrels)', fontsize=14)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('optimal_oil_production.png')
plt.show()

# Plot 2: Target Oil Price over Time
prices = [price_per_barrel(production) for production in optimal_production_profit_old]
plt.figure(figsize=(8, 6))
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