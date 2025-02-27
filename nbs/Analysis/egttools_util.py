import numpy as np
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
from typing import List, Callable, Optional

import egttools as egt

from egttools.plotting.helpers import (xy_to_barycentric_coordinates,
                                       barycentric_to_xy_coordinates,
                                       find_roots_in_discrete_barycentric_coordinates,
                                       calculate_stability,
                                    )
from egttools.analytical.utils import (find_roots, check_replicator_stability_pairwise_games, )
from egttools.helpers.vectorized import vectorized_barycentric_to_xy_coordinates

def replicator_equation(x: np.ndarray, payoff_function: callable, game_params: dict) -> np.ndarray:
    """
    Produces the discrete time derivative of the replicator dynamics with a dynamic payoff matrix.

    This only works for 2-player games.

    Parameters
    ----------
    x : numpy.ndarray[numpy.float64[m,1]]
        array containing the frequency of each strategy in the population.
    payoff_function : callable
        a function that calculates the payoff matrix given the current strategy frequencies and game parameters.
    game_params : dict
        dictionary containing the parameters of the game.

    Returns
    -------
    numpy.ndarray
        time derivative of x

    See Also
    --------
    egttools.analytical.StochDynamics
    egttools.numerical.PairwiseComparisonNumerical
    """
    
    # Calculate dynamic payoffs using the provided function
    payoffs = payoff_function(x, game_params)

    ax = np.dot(payoffs, x)
    x_dot = np.squeeze(x) * (np.squeeze(ax) - np.dot(np.squeeze(x), np.squeeze(ax)))

    return x_dot.reshape(-1, 1)

def vectorized_replicator_equation(frequencies: np.ndarray, payoff_function: callable, game_params: dict) -> np.ndarray:
    """
    Calculate gradients using barycentric coordinates in a strategy space.

    The input `frequencies` is a 3D tensor. The first dimension corresponds to the number of strategies (`p`), 
    and the subsequent `m x n` matrix corresponds to barycentric coordinates in the strategy space. Each row in this 
    matrix represents a specific set of barycentric coordinates, while the columns represent different scenarios 
    or replicates for that set of coordinates.

    Parameters
    ----------
    frequencies: numpy.ndarray[p, m, n]
        A 3D tensor where:
        - `p` is the number of strategies.
        - `m` corresponds to the number of rows of barycentric coordinates.
        - `n` corresponds to the number of columns or replicates of those coordinates.
        
    payoff_function : callable
        A function that calculates the payoff matrix given the current strategy frequencies and game parameters.
        
    game_params : dict
        Dictionary containing the parameters of the game.

    Returns
    -------
    numpy.ndarray[p, m, n]
        The gradients for each set of barycentric coordinates in the strategy space.
    """
    
    p, m, n = frequencies.shape
    gradients = np.zeros((p, m, n))
    for i in range(m):
        for j in range(n):
            current_freq = frequencies[:, i, j]
            # Calculate dynamic payoffs using the provided function
            payoffs = payoff_function(current_freq, game_params)
            
            ax = np.dot(payoffs, current_freq)
            
            # Calculating (ax - x * ax) for the current set of barycentric coordinates
            gradients[:, i, j] = current_freq * (ax - np.sum(current_freq * ax))
        
    return gradients

def check_replicator_stability_pairwise_games(stationary_points: List[np.ndarray],
                                              payoff_function: callable,
                                              game_params: dict,
                                              atol_neg: float = 1e-4,
                                              atol_pos: float = 1e-4,
                                              atol_zero: float = 1e-4) -> List[int]:
    """
    Calculates the stability of the roots assuming that they are from a system governed by the replicator
    equation (this function uses the Jacobian of the replicator equation in pairwise games to calculate the
    stability).

    Parameters
    ----------
    stationary_points: List[numpy.ndarray]
        a list of stationary points (represented as numpy.ndarray).
    payoff_matrix: numpy.ndarray
        a payoff matrix represented as a numpy.ndarray.
    atol_neg: float
        tolerance to consider a value negative.
    atol_pos: float
        tolerance to consider a value positive.
    atol_zero: float
        tolerance to determine if a value is zero.

    Returns
    -------
    List[int]
        A list of integers indicating the stability of the stationary points for the replicator equation:
        1 - stable
        -1 - unstable
        0 - saddle

    """

    def fitness(i: int, x: np.ndarray):
        payoff_matrix = payoff_function(x, game_params)
        return np.dot(payoff_matrix, x)[i]

    # First we build a Jacobian matrix
    def jacobian(x: np.ndarray):
        payoff_matrix = payoff_function(x, game_params)
        ax = np.dot(payoff_matrix, x)
        avg_fitness = np.dot(x, ax)
        jac = [[x[i] * (payoff_matrix[i, j]
                        - np.dot(x, payoff_matrix[:, j]))
                if i != j else (fitness(i, x)
                                - avg_fitness
                                + x[i] * (payoff_matrix[i, i]
                                          - np.dot(x, payoff_matrix[:, i])))
                for i in range(len(x))] for j in range(len(x))]
        return np.asarray(jac)

    stability = []

    for point in stationary_points:
        # now we check the stability of the roots using the jacobian
        eigenvalues = eigvals(jacobian(point))
        # Process eigenvalues to classify those within tolerance as zero
        effective_zero_eigenvalues = [ev for ev in eigenvalues if abs(ev.real) <= atol_zero]
        non_zero_eigenvalues = [ev for ev in eigenvalues if abs(ev.real) > atol_zero]

        print("point: ", point)
        print("eigenvalues: ", eigenvalues)
        print("effective_zero_eigenvalues: ", effective_zero_eigenvalues)
        print("non_zero_eigenvalues: ", non_zero_eigenvalues)
       # If all eigenvalues are effectively zero
        if len(effective_zero_eigenvalues) == len(eigenvalues):
            stability.append(0)  # Marginally or indeterminately stable
        # All non-zero eigenvalues have negative real parts => stable
        elif all(ev.real < -atol_neg for ev in non_zero_eigenvalues):
            stability.append(1)  # Stable
        # All non-zero eigenvalues have positive real parts => unstable
        elif all(ev.real > atol_pos for ev in non_zero_eigenvalues):
            stability.append(-1)  # Unstable
        # Mixture of positive and negative real parts => saddle
        else:
            stability.append(0)  # Saddle

    return stability

# Example payoff function
def example_payoff_function(x: np.ndarray, game_params: dict) -> np.ndarray:
    # This is a placeholder function. For this example, I'm just returning a transposed matrix.
    return np.array([[1, 3], [2, 4]])

# Unit test
def test_replicator_equation():
    x = np.array([[0.5], [0.5]])
    game_params = {}  # Add game parameters here as needed
    
    result = replicator_equation(x, example_payoff_function, game_params)
    expected = np.array([[-0.25], [0.25]])

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

test_replicator_equation()


def test_vectorized_replicator_equation():
    # Define a simple dynamic payoff function
    def sample_payoff_function(freqs: np.ndarray, params: dict) -> np.ndarray:
        # For simplicity, just return a static matrix where the third strategy's payoff 
        # is inversely proportional to its frequency
        return np.array([
            [1, 2, 1 - freqs[2]],
            [2, 3, 2 * (1 - freqs[2])],
            [1, 1, 3]
        ])

    # For this example, let's assume:
    # 3 strategies (for a 2D simplex)
    # 4 elements for each barycentric coordinate
    # 2 barycentric coordinates
    frequencies = np.random.rand(3, 3, 2)  

    # Call the function
    gradients = vectorized_replicator_equation(frequencies, sample_payoff_function, {})

    # Validate the shape of the result
    assert gradients.shape == (3, 3, 2), f"Expected shape (3, 4, 2), but got {gradients.shape}"
    
    def tactical_deception_payoffs(freqs: np.ndarray, params: dict) -> np.ndarray:
        # For simplicity, just return a static matrix where the third strategy's payoff 
        # is inversely proportional to its frequency
        names = ["b", "c", "s", "d"]
        b, c, s, d = [params[k] for k in names]
        q = 1 - (freqs[2] / np.sum(freqs))
        return np.array([
            [b-c, -c*s, -c*(q + s - q*s)],
            [b*s, 0, 0],
            [b*(q + s - q*s) - d, -d, -d]
        ])

    # For this example, let's assume:
    # 3 strategies (for a 2D simplex)
    # 4 elements for each barycentric coordinate
    # 2 barycentric coordinates
    frequencies = np.random.rand(3, 3, 2)  
    params = {"b": 2, "c": 0.5, "d": 0.2, "s": 0.1}
    # Call the function
    gradients = vectorized_replicator_equation(frequencies, tactical_deception_payoffs, params)

    # Validate the shape of the result
    assert gradients.shape == (3, 3, 2), f"Expected shape (3, 3, 2), but got {gradients.shape}"

test_vectorized_replicator_equation()

def tactical_deception_payoffs(freqs: np.ndarray, params: dict) -> np.ndarray:
    # For simplicity, just return a static matrix where the third strategy's payoff 
    # is inversely proportional to its frequency
    names = ["b", "c", "s", "d"]
    b, c, s, d = [params[k] for k in names]
    q = 1 - (freqs[2] / np.sum(freqs))
    payoffs = np.array([
        [b-c, -c*s, -c*(q + s - q*s)],
        [b*s, 0, 0],
        [b*(q + s - q*s) - d, -d, -d]
    ])
    return payoffs

# The following parameters let us replicate the figure.
# params = {"b": 1.5, "c": 0.5, "d": 0.1, "s": 0.2}
# simplex = egt.plotting.Simplex2D()
# frequencies = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
# gradients = vectorized_replicator_equation(frequencies, tactical_deception_payoffs, params)


# xy_results = vectorized_barycentric_to_xy_coordinates(gradients, simplex.corners)
# Ux = xy_results[:, :, 0].astype(np.float64)
# Uy = xy_results[:, :, 1].astype(np.float64)

# calculate_gradients = lambda u: replicator_equation(u, tactical_deception_payoffs, params)[:, 0]
# calculate_gradients_alt = lambda u, t: replicator_equation(u, tactical_deception_payoffs, params)[:, 0]

# roots = find_roots(gradient_function=calculate_gradients,
#                    nb_strategies=frequencies.shape[0],
#                    nb_initial_random_points=100)
# roots_xy = [barycentric_to_xy_coordinates(root, corners=simplex.corners) for root in roots]

# stability = check_replicator_stability_pairwise_games(roots, tactical_deception_payoffs, params)

# # print("stability:", stability)
# type_labels = ['CC', 'HD', 'TD']

# fig, ax = plt.subplots(figsize=(10,8))

# plot = (simplex.add_axis(ax=ax)
#            .apply_simplex_boundaries_to_gradients(Ux, Uy)
#            .draw_triangle()
#            .draw_stationary_points(roots_xy, stability)
#            .add_vertex_labels(type_labels)
#            .draw_scatter_shadow(calculate_gradients_alt, 300, color='gray', marker='.', s=0.1)
#            .draw_gradients()
#          )

# ax.axis('off')
# ax.set_aspect('equal')

# plt.xlim((-.05,1.05))
# plt.ylim((-.02, simplex.top_corner + 0.05))
# plt.show()

# TODO:
# 1. Clean up the new functionality for sending to Elias
# 2. Consider fixing the saddle point code (ask ChatGPT for help too)
# 3. Implement custom replicator equations which allow an implementation
# of higher selection intensities (and mutation rates).
# 4. Consider fixing the Markov Chain code we have.
# 5. Consider modifying Elias' code for Markov Chains.
# 6. Implement a simple ABM for tactical deception as discussed
# 7. Implement an ABM with scale free networks to investigate powerful
# deceptive clusters.
# 8. Plan how we can use LLMs to simulate their chosen strategies and
# social learning.

params = {"b": 1.5, "c": 0.5, "d": 0.1, "s": 0.2}
type_labels = ['CC', 'HD', 'TD']
# calculate_gradients = lambda u: replicator_equation(u, tactical_deception_payoffs, params)[:, 0]
# calculate_gradients_alt = lambda u, t: replicator_equation(u, tactical_deception_payoffs, params)[:, 0]

Z = 100
beta = 1
mu = 1e-3

simplex = egt.plotting.Simplex2D(discrete=True, size=Z, nb_points=Z+1)

frequencies = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))

frequencies_int = np.floor(frequencies * Z).astype(np.int64)

# We make sure that our evolver represents payoffs as our desired function
evolver_payoffs = lambda freqs: tactical_deception_payoffs(freqs, params)

evolver = egt.analytical.StochDynamics(3, evolver_payoffs, Z)

# We also need to ensure that any fitness calculations involving our payoffs
# make use of our own custom methods.

["pop_size", "nb_strategies", "payoffs_fn"]

class_args = {"pop_size": Z,
              "nb_strategies": 3,
              "payoffs_fn": evolver_payoffs}
    
def fitness_pair_functional(x: int, i: int, j: int, *args: Optional[list]) -> float:
        """
        Calculates the fitness of strategy i versus strategy j, in
        a population of x i-strategists and (pop_size-x) j strategists, considering
        a 2-player game.

        Parameters
        ----------
        x : int
            number of i-strategists in the population
        i : int
            index of strategy i
        j : int
            index of strategy j
        args : Optional[list]

        Returns
        -------
            float
            the fitness difference among the strategies
        """
        names = ["pop_size", "nb_strategies", "payoffs_fn"]
        pop_size, nb_strategies, payoffs_fn = [class_args[k] for k in names]
        popoulation_state_dict = {i: x, j: pop_size - x}
        population_state = [popoulation_state_dict.get(k, 0)
                            for k in range(nb_strategies)]
        payoff_matrix = payoffs_fn(population_state)
        fitness_i = ((x - 1) * payoff_matrix[i, i] +
                    (pop_size - x) * payoff_matrix[i, j]) / (pop_size - 1)
        fitness_j = ((pop_size - x - 1) * payoff_matrix[j, j] +
                    x * payoff_matrix[j, i]) / (pop_size - 1)
        return fitness_i - fitness_j

def full_fitness_difference_pairwise_functional(i: int, j: int, population_state: np.ndarray) -> float:
        """
        Calculates the fitness of strategy i in a population with state :param population_state,
        assuming pairwise interactions (2-player game).

        Parameters
        ----------
        i : int
            index of the strategy that will reproduce
        j : int
            index of the strategy that will die
        population_state : numpy.ndarray[numpy.int64[m,1]]
                        vector containing the counts of each strategy in the population

        Returns
        -------
        float
        The fitness difference between the two strategies for the given population state
        """
        names = ["pop_size", "nb_strategies", "payoffs_fn"]
        pop_size, nb_strategies, payoffs_fn = [class_args[k] for k in names]
        # Here, our payoffs depend on the population state.
        payoff_matrix = payoffs_fn(population_state)
        fitness_i = (population_state[i] - 1) * payoff_matrix[i, i]
        for strategy in range(nb_strategies):
            if strategy == i:
                continue
            fitness_i += population_state[strategy] * payoff_matrix[i, strategy]
        fitness_j = (population_state[j] - 1) * payoff_matrix[j, j]
        for strategy in range(nb_strategies):
            if strategy == j:
                continue
            fitness_j += population_state[strategy] * payoff_matrix[j, strategy]

        return (fitness_i - fitness_j) / (pop_size - 1)

evolver.full_fitness = full_fitness_difference_pairwise_functional
evolver.fitness = fitness_pair_functional

result = np.asarray([[evolver.full_gradient_selection(frequencies_int[:, i, j], beta)
                      for j in range(frequencies_int.shape[2])]
                     for i in range(frequencies_int.shape[1])]).swapaxes(0, 1).swapaxes(0, 2)

xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)
Ux = xy_results[:, :, 0].astype(np.float64)
Uy = xy_results[:, :, 1].astype(np.float64)

calculate_gradients = lambda u: Z*evolver.full_gradient_selection(u, beta)

roots = find_roots_in_discrete_barycentric_coordinates(calculate_gradients, Z, nb_interior_points=5151, atol=1e-1)
roots_xy = [barycentric_to_xy_coordinates(x, simplex.corners) for x in roots]

stability = calculate_stability(roots, calculate_gradients)

evolver.mu = 0
sd_rare_mutations = evolver.calculate_stationary_distribution(beta)
print("Stationary Distribution of the 3 strategies: ", sd_rare_mutations)

evolver.mu = 1e-3
sd = evolver.calculate_stationary_distribution(beta)

fig, ax = plt.subplots(figsize=(15,10))

plot = (simplex.add_axis(ax=ax)
           .apply_simplex_boundaries_to_gradients(Ux, Uy)
           .draw_gradients(zorder=5)
           .add_colorbar()
           .draw_stationary_points(roots_xy, stability, zorder=11)
           .add_vertex_labels(type_labels)
           .draw_stationary_distribution(sd, vmax=0.0001, alpha=0.5, edgecolors='gray', cmap='binary', shading='gouraud', zorder=0)
         )

ax.axis('off')
ax.set_aspect('equal')

plt.xlim((-.05,1.05))
plt.ylim((-.02, simplex.top_corner + 0.05))
plt.show()
