
import random
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    for _ in range(0, size_of_generation-1):
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



def show_solution_with_fitness(solutions):
    for solution, fitness_score in solutions:
        print(f"Fitness score: {fitness_score}", end=' ')
        airplane_indexes = [index for index, airplane in solution]
        print(airplane_indexes)


def show_solution_airplace_indexes(solution):
    airplane_indexes = [index for index, airplane in solution]
    print(airplane_indexes)


def add_fitness_to_generation(generation):
    return [(solution, fitness_function(solution)) for solution in generation]


def get_metrics_from_generation(generation):
    fitness_scores = [fitness for _, fitness in generation]
    return {
        'max': max(fitness_scores),
        'min': min(fitness_scores),
        'mean': sum(fitness_scores) / len(fitness_scores)
    }


def tournament_match(participants):
    # Winnes the parent with the lowest fitness score
    p1, p2 = participants

    if p1[1] <= p2[1]:
        return p1
    return p2


def generate_participants(generation):
    # Shuffle the input array in-place
    random.shuffle(generation)

    # Create pairs of consecutive elements
    participants = [(generation[i], generation[i + 1]) for i in range(0, len(generation) - 1, 2)]

    return participants


def perform_crossover(tournament_winners, i):
    crossover_point = random.randint(1, len(tournament_winners[i][0])-2)
    parent_1 = tournament_winners[i][0]
    parent_2 = tournament_winners[i + 1][0]
    child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
    child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    # print("Crossover point:", crossover_point)
    # print("Parent 1: ")
    # show_solution_airplace_indexes(parent_1)
    # print("Parent 2: ")
    # show_solution_airplace_indexes(parent_2)
    # print("Child 1: ")
    # show_solution_airplace_indexes(child_1)
    # print("Child 2: ")
    # show_solution_airplace_indexes(child_2)

    return (parent_1, parent_2, child_1, child_2)


def perform_mutation(mutation_rate, child):
    # Mutation
    if random.random() < mutation_rate:

        mutation_point1 = random.randint(0, len(child) - 1)
        mutation_point2 = random.randint(0, len(child) - 1)

        while mutation_point1 == mutation_point2:
            mutation_point2 = random.randint(0, len(child) - 1)

        aux = child[mutation_point1]
        child[mutation_point1] = child[mutation_point2]
        child[mutation_point2] = aux
    return child

def show_fitness_evolution_animation(metrics):
    x_vec = list(range(1, number_of_generations + 1))
    y_vec = metrics['mean']

    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Best Fitness Score Evolution')
    title = ax.set_title('Average Fitness Score Evolution')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness Score')

    # Set the x-axis range
    ax.set_xlim(0, number_of_generations)

    # Set the y-axis range
    ax.set_ylim(0, max(y_vec))

    def update(frame):
        line.set_data(x_vec[:frame], y_vec[:frame])
        title.set_text(f'Genetic Algorithm\nGeneration: {frame+1} - AvgFitness: {y_vec[frame]:.2f} - BestFitness: {metrics["min"][frame]:.2f}')

        fig.canvas.draw()
        fig.canvas.flush_events()
        return line,

    ani = FuncAnimation(fig, update, frames=len(x_vec), interval=0, blit=True, repeat=False)
    plt.legend()
    plt.show()


def genetic_algorithm(generation_with_fitness, mutation_rate):
    # Tournament selection
    tournament_winners = []
    next_generation = []

    # The participants are pairs of two of solutions
    participants = generate_participants(generation_with_fitness)

    # If there is not enought solutions for the tournament, add the last solution to the next generation
    if len(generation_with_fitness) % 2 == 1:
        next_generation.append(generation_with_fitness[-1])

    # For each pair of solutions, perform a tournament match and get the winner
    for i in range(0, len(participants)):
        tournament_winners.append(tournament_match(participants[i]))
    # Tournament selection ends

    # Crossover and mutation
    for i in range(0, len(tournament_winners), 2):
        # If there is an odd number of elements, handle by adding the first again to the next generation
        if (i + 1) >= len(tournament_winners):
            next_generation.append(tournament_winners[i])
            next_generation.append(tournament_winners[0])
            break

        # Crossover
        parent_1, parent_2, child_1, child_2 = perform_crossover(tournament_winners, i)


        # Mutation
        child_1 = perform_mutation(mutation_rate, child_1)
        child_2 = perform_mutation(mutation_rate, child_2)

        next_generation.append(parent_1)
        next_generation.append(parent_2)
        next_generation.append(child_1)
        next_generation.append(child_2)

    return next_generation


## To identify the airplanes, we can use the index of the list
airplane_stream = generate_airplane_stream(200)

# Good baseline for the algorithms to start with
# Sort the airplane_stream by expected landing time
sorted_airplane_stream = sorted(airplane_stream, key=lambda x: x[1].expected_landing_time)


first_generation = generate_possible_solutions(sorted_airplane_stream, 200)
print("Possible solutions:", len(first_generation))

generation_with_fitness = add_fitness_to_generation(first_generation)

generation = genetic_algorithm(generation_with_fitness, 0.01)
metrics = {'min' : [], 'max' : [], 'mean' : []}
number_of_generations = 500

for i in range(0, number_of_generations):
    generation_with_fitness = add_fitness_to_generation(generation)
    # show_solution_with_fitness(generation_with_fitness)
    metric = get_metrics_from_generation(generation_with_fitness)
    generation = genetic_algorithm(generation_with_fitness, 0.01)

    # Add the metric to the list of metrics
    metrics['min'].append(metric['min'])
    metrics['mean'].append(metric['mean'])

    # Print the metrics
    print("Generation", i+1)
    print("Best Fitness:", metric['min'])
    print("Average Fitness:", metric['mean'])

show_fitness_evolution_animation(metrics)
