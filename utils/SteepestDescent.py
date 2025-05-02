import time
from copy import deepcopy

from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult


class SteepestDescent(Optimizer):
    """Riemannian steepest descent algorithm.

    Perform optimization using gradient descent with line search.
    This method first computes the gradient of the objective, and then
    optimizes by moving in the direction of steepest descent (which is the
    opposite direction to the gradient).

    Args:
        line_searcher: The line search method.
    """

    def __init__(self, line_searcher=None, num_iter=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if line_searcher is None:
            self._line_searcher = BackTrackingLineSearcher()
        else:
            self._line_searcher = line_searcher
        self.line_searcher = None
        self._num_iter = num_iter 

    # Function to solve optimisation problem using steepest descent.
    def run(
        self, problem, *, initial_point=None, reuse_line_searcher=False
    ) -> OptimizerResult:
        """Run steepest descent algorithm.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            reuse_line_searcher: Whether to reuse the previous line searcher.
                Allows to use information from a previous call to
                :meth:`solve`.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        manifold = problem.manifold
        objective = problem.cost
        gradient = problem.riemannian_gradient

        if not reuse_line_searcher or self.line_searcher is None:
            self.line_searcher = deepcopy(self._line_searcher)
        line_searcher = self.line_searcher

        # If no starting point is specified, generate one at random.
        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point

        self._initialize_log(
            optimizer_parameters={"line_searcher": line_searcher}
        )

        # Initialize iteration counter and timer
        iteration = 0
        start_time = time.time()

        for iteration in range(self._num_iter):

            # Calculate new cost, grad and gradient_norm
            cost = objective(x)
            grad = gradient(x)
            gradient_norm = manifold.norm(x, grad)

            desc_dir = -grad

            # Perform line-search
            step_size, x = line_searcher.search(
                objective, manifold, x, desc_dir, cost, -(gradient_norm**2)
            )

        return self._return_result(
            start_time=start_time,
            point=x,
            cost=objective(x),
            iterations=iteration,
            stopping_criterion=None,
            cost_evaluations=iteration,
            step_size=step_size,
            gradient_norm=gradient_norm,
        )
