import random
import pandas as pd

class Airplane:
    def __init__(self):
        # Randomly generate fuel level between 1000 and 5000 liters
        self.arriving_fuel_level = random.uniform(1000, 5000)
        # Randomly generate fuel consumption rate between 5 and 20 liters per minute
        self.fuel_consumption_rate = random.uniform(5, 20)
        # Randomly generate expected landing time between 10 and 120 minutes from now
        self.expected_landing_time = random.uniform(10, 120)

def generate_airplane_stream(num_airplanes):
    airplane_stream = [(index, Airplane()) for index in range(num_airplanes)]
    return airplane_stream

def generate_single_solution(airplane_stream):
    shuffled_airplane_stream = airplane_stream.copy()
    random.shuffle(shuffled_airplane_stream)
    return shuffled_airplane_stream

## Generate n_solutions possible solutions
def generate_possible_solutions(sorted_airplane_stream, size_of_generation):
    solutions = []
    solutions.append(sorted_airplane_stream)

    for _ in range(0, size_of_generation):
        solutions.append(generate_single_solution(sorted_airplane_stream))

    return solutions

def check_fuel_related_incident(airplane, current_time):
    consumed_fuel = airplane.fuel_consumption_rate * current_time
    current_fuel_level = airplane.arriving_fuel_level - consumed_fuel
    return not(current_fuel_level >= 60 * airplane.fuel_consumption_rate)

def get_fuel_time_left(airplane, current_time):
    consumed_fuel = airplane.fuel_consumption_rate * current_time
    current_fuel = airplane.arriving_fuel_level - consumed_fuel
    return current_fuel / airplane.fuel_consumption_rate

def check_for_crash(airplane, current_time):
    consumed_fuel = airplane.fuel_consumption_rate * current_time
    current_fuel_level = airplane.arriving_fuel_level - consumed_fuel
    return current_fuel_level <= 0

def fitness_function(airplane_stream):
    current_time = 0
    reverse_fitness_score = 0

    # Process the planes in groups of three
    # For each 3 planes we add 3 minutes to the current time
    for i in range(0, len(airplane_stream), 3):
        # Get the group of 3 planes with error safety check
        group = airplane_stream[i:] if (i + 3) > len(airplane_stream) else airplane_stream[i:i+3]

        # Process each plane in the group
        for (_, airplane) in group:
            # Check if the plane has a crash
            crash_weight = 0
            if check_for_crash(airplane, current_time):
                crash_weight = 100

            # Check if the plane has a fuel related incident
            fuel_incident_weight = 0
            if check_fuel_related_incident(airplane, current_time):
                # In case of accident, subtract the remaining fuel time multiplied by 0.02, because the greater the remaining fuel time, the better
                fuel_incident_weight = 2 - (get_fuel_time_left(airplane, current_time) * 0.02)

            # If the plane is late, add the difference between the current time and the expected landing time multiplied by 0.01
            expected_landing_time_weight = 0
            if current_time > airplane.expected_landing_time:
                expected_landing_time_weight = (current_time - airplane.expected_landing_time) * 0.01

            reverse_fitness_score += fuel_incident_weight + expected_landing_time_weight + crash_weight

        # Add 3 minutes to the current time
        current_time += 3

    return reverse_fitness_score

## To identify the airplanes, we can use the index of the list
airplane_stream = generate_airplane_stream(100)

# Good baseline for the algorithms to start with
# Sort the airplane_stream by expected landing time
sorted_airplane_stream = sorted(airplane_stream, key=lambda x: x[1].expected_landing_time)

possible_solutions = generate_possible_solutions(sorted_airplane_stream, 2000)

# Create an empty list to store the DataFrames of each solution
dfs = []

for solution in possible_solutions:
    # Create the dataframe
    df = pd.DataFrame()
    df["Airplane number"] = [element[0] for element in solution]
    df["Fuel consumption rate"] = [element[1].fuel_consumption_rate for element in solution]
    df["Arriving fuel level"] = [element[1].arriving_fuel_level for element in solution]
    df["Expected landing time"] = [element[1].expected_landing_time for element in solution]

    # Change the index to "Airplane Number"
    df.set_index("Airplane number", inplace=True)

    # Add the fitness score column to the DataFrame
    df["Fitness score"] = fitness_function(solution)

    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all the DataFrames in the list
result_df = pd.concat(dfs).sort_values(by="Fitness score", ascending=True).round(2)

print(result_df)




