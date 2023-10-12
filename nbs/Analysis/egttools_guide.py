import numpy as np
import matplotlib.pyplot as plt

import egttools as egt

from egttools.plotting.helpers import (xy_to_barycentric_coordinates,
                                       barycentric_to_xy_coordinates,
                                       find_roots_in_discrete_barycentric_coordinates,
                                       calculate_stability,
                                    )
from egttools.analytical.utils import (find_roots, check_replicator_stability_pairwise_games, )
from egttools.helpers.vectorized import vectorized_replicator_equation, vectorized_barycentric_to_xy_coordinates


payoffs = np.array([[1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 3]])

simplex = egt.plotting.Simplex2D()

v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))

results = vectorized_replicator_equation(v, payoffs)
xy_results = vectorized_barycentric_to_xy_coordinates(results, simplex.corners)
Ux = xy_results[:, :, 0].astype(np.float64)
Uy = xy_results[:, :, 1].astype(np.float64)

calculate_gradients = lambda u: egt.analytical.replicator_equation(u, payoffs)

roots = find_roots(gradient_function=calculate_gradients,
                   nb_strategies=payoffs.shape[0],
                   nb_initial_random_points=100)
roots_xy = [barycentric_to_xy_coordinates(root, corners=simplex.corners) for root in roots]

stability = check_replicator_stability_pairwise_games(roots, payoffs)

type_labels = ['A', 'B', 'C']

fig, ax = plt.subplots(figsize=(10,8))

plot = (simplex.add_axis(ax=ax)
           .apply_simplex_boundaries_to_gradients(Ux, Uy)
           .draw_triangle()
           .draw_stationary_points(roots_xy, stability)
           .add_vertex_labels(type_labels)
           .draw_trajectory_from_roots(lambda u, t: egt.analytical.replicator_equation(u, payoffs),
                                       roots,
                                       stability,
                                       trajectory_length=15,
                                       linewidth=1,
                                       step=0.01,
                                       color='k', draw_arrow=True, arrowdirection='right', arrowsize=30, zorder=4, arrowstyle='fancy')
           .draw_scatter_shadow(lambda u, t: egt.analytical.replicator_equation(u, payoffs), 300, color='gray', marker='.', s=0.1)
          )

ax.axis('off')
ax.set_aspect('equal')

plt.xlim((-.05,1.05))
plt.ylim((-.02, simplex.top_corner + 0.05))
plt.show()

