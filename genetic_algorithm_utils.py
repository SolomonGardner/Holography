import numpy as np


def mutate_binary(child, mutation_rate):
    mutation_mask = np.random.rand(*child.shape) < mutation_rate
    # Perform mutation: Flip bits where mutation_mask is True
    child = np.where(mutation_mask, 1 - child, child)

    return child


def mutate(child: np.ndarray,
                      mutation_rate: float,
                      sigma: float = 0.05,
                      mode: str = "add"):

    mutation_mask = np.random.rand(*child.shape) < mutation_rate

    if mode == "add":
        delta = np.random.normal(loc=0.0, scale=sigma, size=child.shape)
        child = (child + mutation_mask * delta) % 1.0      # wrap-around

    elif mode == "replace":
        new_vals = np.random.rand(*child.shape)
        child = np.where(mutation_mask, new_vals, child)

    else:
        raise ValueError("mode must be 'add' or 'replace'")

    return child.astype(np.float32)


def create_children_npc_elitism(num_best_parents, mutation_rate, total_population_goal=1000):
    def crossover_function(H_parents, n_splits):
        # Start the new generation with the first num_best_parents from H_parents
        new_generation = H_parents[:num_best_parents]  # Assuming H_parents are sorted by their fitness

        # Calculate the total number of elements in a parent matrix
        num_elements = H_parents[0].size  # Assuming all parents have the same shape

        # Generate new children until the total population goal is reached
        while len(new_generation) < total_population_goal:
            # Randomly select two parents from the entire original parent set, not just the top ones
            parents_indices = np.random.choice(len(H_parents), 2, replace=False)
            parent1, parent2 = H_parents[parents_indices[0]], H_parents[parents_indices[1]]

            # Generate n_splits crossover points
            crossover_points = np.sort(np.random.randint(1, num_elements - 1, size=n_splits))

            # Perform n-point crossover
            child_flattened = np.empty(num_elements)
            parent1_flattened = parent1.flatten()
            parent2_flattened = parent2.flatten()

            # Alternate segments from each parent based on the crossover points
            last_point = 0
            take_from_parent1 = True
            for point in crossover_points:
                if take_from_parent1:
                    child_flattened[last_point:point] = parent1_flattened[last_point:point]
                else:
                    child_flattened[last_point:point] = parent2_flattened[last_point:point]
                take_from_parent1 = not take_from_parent1
                last_point = point

            # Fill in the final segment after the last crossover point
            if take_from_parent1:
                child_flattened[last_point:] = parent1_flattened[last_point:]
            else:
                child_flattened[last_point:] = parent2_flattened[last_point:]

            # Reshape back to the original matrix shape and apply mutation
            child = child_flattened.reshape(parent1.shape)
            child = mutate(child, mutation_rate)  # Assuming mutate function is defined elsewhere

            # Append the mutated child to the new generation
            new_generation.append(child)

        return new_generation
    return crossover_function


def select_parents_tournament(num_parents=500, tournament_size=10):
    def parent_selection_function(mse_list, H_list):
        selected_parents = []

        for _ in range(num_parents):
            # Randomly select tournament_size individuals for the tournament
            tournament_indices = np.random.choice(len(mse_list), tournament_size, replace=False)
            tournament_mses = [mse_list[i] for i in tournament_indices]

            # Find the index of the individual with the lowest MSE in the tournament
            winner_index = tournament_indices[np.argmin(tournament_mses)]

            # Add the winner to the list of selected parents
            selected_parents.append(H_list[winner_index])

        return selected_parents
    return parent_selection_function


def select_parents_roulette(num_parents=500):
    def parent_selection_function(mse_list, H_list):
        # Convert mse_list to a NumPy array if it's not already one
        mse_array = np.array(mse_list)

        # Invert MSE scores to handle minimization problem (higher is better for selection)
        fitness_scores = 1 / (mse_array)

        # Normalize fitness scores to sum to 1
        probabilities = fitness_scores / np.sum(fitness_scores)

        # Select parents using roulette wheel selection
        parent_indices = np.random.choice(range(len(mse_array)), size=num_parents, p=probabilities, replace=True)
        selected_parents = [H_list[i] for i in parent_indices]

        return selected_parents
    return parent_selection_function
