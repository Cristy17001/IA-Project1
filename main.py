import random
import json
import math


class Airplane:
    def __init__(self):
        # Randomly generate fuel level between 1000 and 5000 liters
        self.arriving_fuel_level = random.uniform(1000, 5000)
        # Randomly generate fuel consumption rate between 5 and 20 liters per minute
        self.fuel_consumption_rate = random.uniform(5, 20)
        # Randomly generate expected landing time between 10 and 120 minutes from now
        self.expected_landing_time = random.uniform(10, 120)

report_data = {'genetic': {}, 'simulated_annealing': {}, 'first_airplane_stream': {}}

def generate_airplane_stream(num_airplanes):
    airplane_stream = [(index, Airplane()) for index in range(num_airplanes)]
    for index, airplane in airplane_stream:
        report_data['first_airplane_stream']['index'] = [index for index, _ in airplane_stream]
        report_data['first_airplane_stream']['arriving_fuel_level'] = [airplane.arriving_fuel_level for _, airplane in airplane_stream]
        report_data['first_airplane_stream']['fuel_consumption_rate'] = [airplane.fuel_consumption_rate for _, airplane in airplane_stream]
        report_data['first_airplane_stream']['expected_landing_time'] = [airplane.expected_landing_time for _, airplane in airplane_stream]

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
    min_solution = [solution for solution, fitness in generation if fitness == min(fitness_scores)]
    airplane_indexes = [index for index, airplane in min_solution[0]]
    return {
        'min': min(fitness_scores),
        'mean': sum(fitness_scores) / len(fitness_scores),
        'min_solution': airplane_indexes
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

def generate_single_solution2(airplane_stream):
    # Make a copy of the airplane stream
    new_solution = airplane_stream.copy()

    # Select two random indexes within the range of the solution
    idx1, idx2 = random.sample(range(len(new_solution)), 2)

    # Swap the positions of the airplanes at the selected indexes
    new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]

    return new_solution

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


def simulated_annealing_with_tracking(initial_solution, temperature, cooling_rate, num_iterations):
    current_solution = initial_solution
    current_fitness = fitness_function(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness

    all_solutions = [current_solution]

    for _ in range(num_iterations):
        # Generate a neighboring solution
        next_solution = generate_single_solution2(current_solution)

        # Calculate fitness for the neighboring solution
        next_fitness = fitness_function(next_solution)

        # Calculate energy difference
        energy_difference = next_fitness - current_fitness

        # Accept the new solution if it's better or with a certain probability if it's worse
        if energy_difference < 0 or random.random() < acceptance_probability(energy_difference, temperature):
            current_solution = next_solution
            current_fitness = next_fitness

            # Update the best solution if needed
            if current_fitness < best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness


        # Cool down the temperature
        temperature *= cooling_rate

        all_solutions.append(current_solution)

    return best_solution, best_fitness, all_solutions

def acceptance_probability(energy_difference, temperature):
    if energy_difference < 0:
        return 1
    else:
        if temperature == 0:
            return 0
        return math.exp(-energy_difference / temperature)

def generate_report(report_data):
    # Convert the report data to JSON format
    report_json = json.dumps(report_data)

    # Specify the file path to save the report
    file_path = "report.json"

    # Write the report data to the file
    with open(file_path, "w") as file:
        file.write(report_json)

    print("Report generated and saved to", file_path)

def menu():
    print("Choose an algorithm:")
    print("1. Genetic Algorithm")
    print("2. Simulated Annealing")
    print("3. Full Report")

    choice = input("Enter your choice (1 or 2 or 3): ")
    airplane_stream = generate_airplane_stream(200)

    if choice == "1":
        number_of_generations, mutation_rate = ask_genetic_algorithm_parameters()
        run_genetic_algorithm(airplane_stream, number_of_generations, mutation_rate)
        generate_report(report_data)
    elif choice == "2":
        num_iterations, initial_temperature, cooling_rate = ask_simulated_annealing_parameters()
        run_simulated_annealing(airplane_stream, num_iterations, initial_temperature, cooling_rate)
        generate_report(report_data)
    elif choice == "3":
        number_of_generations, mutation_rate = ask_genetic_algorithm_parameters()
        num_iterations, initial_temperature, cooling_rate = ask_simulated_annealing_parameters()

        run_genetic_algorithm(airplane_stream, number_of_generations, mutation_rate)
        run_simulated_annealing(airplane_stream, num_iterations, initial_temperature, cooling_rate)
        generate_report(report_data)
    else:
        print("Invalid choice. Please enter 1 or 2.")
        menu()

def ask_genetic_algorithm_parameters():
    print("====================================")
    print("\tGenetic Algorithm:")
    print("====================================")
    number_of_generations = int(input("Enter the number of generations: "))
    mutation_rate = float(input("Enter the mutation rate (between 0 and 1): "))

    return number_of_generations, mutation_rate

def ask_simulated_annealing_parameters():
    print("====================================")
    print("\tSimulated annealing:")
    print("====================================")
    num_iterations = int(input("Enter the number of iterations: "))
    initial_temperature = float(input("Enter the initial temperature: "))
    cooling_rate = float(input("Enter the cooling rate (between 0 and 1): "))

    return num_iterations, initial_temperature, cooling_rate


def run_genetic_algorithm(airplane_stream, number_of_generations, mutation_rate):
    sorted_airplane_stream = sorted(airplane_stream, key=lambda x: x[1].expected_landing_time)
    print("====================================")
    print("    Running Genetic Algorithm:")
    print("====================================")
    first_generation = generate_possible_solutions(sorted_airplane_stream, 200)
    generation_with_fitness = add_fitness_to_generation(first_generation)

    generation = genetic_algorithm(generation_with_fitness, mutation_rate)
    metrics = {'min': [], 'mean': [], 'min_solution': []}

    # Validate the mutation rate
    if mutation_rate < 0 or mutation_rate > 1:
        print("Invalid mutation rate. Please enter a value between 0 and 1.")
        menu()

    for i in range(number_of_generations):
        generation_with_fitness = add_fitness_to_generation(generation)
        metric = get_metrics_from_generation(generation_with_fitness)
        generation = genetic_algorithm(generation_with_fitness, mutation_rate)
        metrics['min'].append(metric['min'])
        metrics['mean'].append(metric['mean'])
        metrics['min_solution'].append(metric['min_solution'])

        print("Generation", i + 1)
        print("Best Fitness:", metric['min'])
        # print("Best Solution:", metric['min_solution'])
        print("Average Fitness:", metric['mean'])

    report_data['genetic'] = metrics


def show_all_solutions(all_solutions):
    for i, solution in enumerate(all_solutions):
        print(f"Iteration {i + 1}:")
        show_solution_airplace_indexes(solution)

def run_simulated_annealing(airplane_stream, num_iterations, initial_temperature, cooling_rate):
    sorted_airplane_stream = sorted(airplane_stream, key=lambda x: x[1].expected_landing_time)
    initial_solution = generate_single_solution(sorted_airplane_stream)
    print("====================================")
    print("    Running Simulated annealing:")
    print("====================================")
    best_solution, best_fitness, all_solutions = simulated_annealing_with_tracking(initial_solution, initial_temperature, cooling_rate, num_iterations)

    # Show iteration number and fitness score
    print("Iterations and Fitness Scores:")
    iterations = []
    fitness_scores = []
    solutions = []
    for i, solution in enumerate(all_solutions):
        fitness_score = fitness_function(solution)
        iterations.append(i)
        fitness_scores.append(fitness_score)
        solutions.append([i for i, _ in solution])
        # show_solution_airplace_indexes(solution)
        print(f"Iteration {i + 1}: Best Fitness Score: {fitness_score}")
    report_data['simulated_annealing'] = {'iterations': iterations, 'fitness_scores': fitness_scores, 'solutions': solutions}

# Call the menu function to start
if __name__ == "__main__":
    menu()