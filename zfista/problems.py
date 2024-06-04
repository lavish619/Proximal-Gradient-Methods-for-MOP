import warnings
from abc import abstractmethod
from functools import cmp_to_key
from typing import List, Optional, Sequence, Tuple, Union

import jax
import numpy as np
from jaxopt.projection import projection_box, projection_simplex, projection_non_negative
from jaxopt.prox import prox_lasso
from scipy.optimize import OptimizeResult, root_scalar
from scipy import sparse

from zfista import minimize_proximal_gradient

jax.config.update("jax_enable_x64", True)


class Problem:
    """Superclass of test problems to be solved by the proximal gradient methods for
    multiobjective optimization.

    In all test problems, each objective function can be written as

    .. math::

        F_i(x) = f_i(x) + g_i(x),

    where :math:`f_i` is convex and differentiable and :math:`g_i` is closed, proper and convex.

    Parameters
    ----------
    n_features
        The dimension of the decision variable.

    n_objectives
        The number of objective functions.

    l1_ratios
        An array of shape (n_objectives,) containing the coefficients for the L1 regularization term for each objective function.
        If not provided, no L1 regularization is applied.

    l1_shifts
        An array of shape (n_objectives,) containing the shifts for the L1 regularization term for each objective function.
        If not provided, no shifts are applied.

    bounds
        A tuple with two elements representing the lower and upper bounds of the decision variable.
        Each element can be a scalar or an array of shape (n_features,).
        If not provided, no bounds are applied.
    """

    def __init__(
        self,
        n_features: int,
        n_objectives: int,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        self.n_features = n_features
        self.n_objectives = n_objectives
        self.l1_ratios = l1_ratios
        if l1_ratios is not None:
            self.l1_shifts = (
                np.zeros(n_objectives) if l1_shifts is None else np.array(l1_shifts)
            )
        self.bounds = bounds
        self.name = self._generate_name()
        self.problem_name = type(self).__name__

    def _generate_name(self) -> str:
        name_parts = [type(self).__name__, f"n_{self.n_features}"]
        self.g_name = "Zero function"
        if self.l1_ratios is not None:
            l1_ratios_str = "_".join(map(str, self.l1_ratios))
            name_parts.append(f"l1_ratios_{l1_ratios_str}")
            l1_shifts_str = "_".join(map(str, self.l1_shifts))
            name_parts.append(f"l1_shifts_{l1_shifts_str}")
            self.g_name = f"$l_1$ function"
        if self.bounds is not None:
            bounds_str = "_".join(map(str, [self.bounds[0], self.bounds[1]]))
            name_parts.append(f"bounds_{bounds_str}")
            self.g_name = f"Indicator function"
        return "_".join(name_parts)

    @abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        pass

    def g(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        if self.bounds is not None:
            if (x < self.bounds[0]).any() or (x > self.bounds[1]).any():
                return np.full(self.n_objectives, np.inf)
        if self.l1_ratios is not None:
            if self.n_objectives != len(self.l1_ratios):
                raise ValueError("len(l1_ratios) should be equal to n_objectives.")
            if self.n_objectives != len(self.l1_shifts):
                raise ValueError("len(l1_shifts) should be equal to n_objectives.")
            return self.l1_ratios * np.linalg.norm(
                x - self.l1_shifts.reshape(-1, 1), ord=1, axis=1
            )
         
        if self.problem_name == "SCAD":
            return self.g_x(x) 
        
        if self.problem_name == "NonConvex_Quadratic":
            if abs(np.sum(x) - self.s) > 1e-3:
                return np.full(self.n_objectives, np.inf)

        return np.zeros(self.n_objectives)

    def prox_wsum_g(self, weight: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        if self.n_objectives != len(weight):
            raise ValueError("len(weight) should be equal to n_objectives.")
        if self.l1_ratios is not None:
            coef = weight * self.l1_ratios
            x = prox_lasso(
                x + np.sum(coef[1:]) - self.l1_shifts[0] + self.l1_shifts[0], coef[0]
            )
            for i in range(1, self.n_objectives):
                x = (
                    prox_lasso(x - coef[i] - self.l1_shifts[i], coef[i])
                    + self.l1_shifts[i]
                )
        if self.bounds is not None:
            x = projection_box(x, (self.bounds[0], self.bounds[1]))
        
        if self.problem_name == "NonConvex_Quadratic":
            t = np.random.uniform(0, 1, 1)
            x = projection_simplex(x, max(1, 10*t))
        return x

    def minimize_proximal_gradient(self, x0: np.ndarray, **kwargs) -> OptimizeResult:
        return minimize_proximal_gradient(
            self.f,
            self.g,
            self.jac_f,
            self.prox_wsum_g,
            x0,
            **kwargs,
        )


class JOS1(Problem):
    r"""n_features = 5 (default), n_objectives = 2

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = (1 / n) \sum_i x_i^2, \\
        f_2(x) = (1 / n) \sum_i (x_i - 2)^2.
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \nabla f_1(x) = (2 / n) x, \nabla f_2(x) = (2 / n) (x - 2).

    Reference: Jin, Y., Olhofer, M., Sendhoff, B.: Dynamic weighted aggregation for evolutionary multi-objective optimization: Why does it work and how? In: GECCO’01 Proceedings of the 3rd Annual Conference on Genetic and Evolutionary Computation, pp. 1042–1049 (2001)
    """

    def __init__(
        self,
        n_features: int = 2,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = np.linalg.norm(x) ** 2 / self.n_features
        f2 = np.linalg.norm(x - 2) ** 2 / self.n_features
        return np.array([f1, f2])

    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        jac_f1 = 2 * x / self.n_features
        jac_f2 = 2 * (x - 2) / self.n_features
        return np.vstack((jac_f1, jac_f2))


class SD(Problem):
    r"""n_features = 4, n_objectives = 2

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = 2 x_1 + \sqrt{2} x_2 + \sqrt{2} x_3 + x_4, \\
        f_2(x) = 2 / x_1 + 2 \sqrt{2} / x_2 + 2 \sqrt{2} / x_3 + x_4,
        \end{gathered}

    subject to

    .. math::

        [1, \sqrt{2}, \sqrt{2}, 1] \le x \le [3, 3, 3, 3].

    Each gradient of f_i can be written as

    .. math::

        \nabla f_1(x) = [1, \sqrt{2}, \sqrt{2}, 1], \nabla f_2(x) = 0.

    Reference: Stadler, W., Dauer, J.: Multicriteria optimization in engineering: a tutorial and survey. In: Kamat, M.P. (ed.) Progress in Aeronautics and Astronautics: Structural Optimization: Status and Promise, vol. 150, pp. 209–249. American Institute of Aeronautics and Astronautics, Reston (1992)
    """

    def __init__(
        self,
        n_features: int = 4,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,

    ):
        super().__init__(
            n_features=4,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = 2 * x[0] + np.sqrt(2) * x[1] + np.sqrt(2) * x[2] + x[3]
        f2 = 2 / x[0] + 2 * np.sqrt(2) / x[1] + 2 * np.sqrt(2) / x[2] + 2 / x[3]
        return np.array([f1, f2])

    def jac_f(self, x: np.ndarray) -> np.ndarray:
        jac_f1 = np.array([2, np.sqrt(2), np.sqrt(2), 1])
        jac_f2 = np.array(
            [
                -2 / x[0] ** 2,
                -2 * np.sqrt(2) / x[1] ** 2,
                -2 * np.sqrt(2) / x[2] ** 2,
                -2 / x[3] ** 2,
            ]
        )
        return np.vstack((jac_f1, jac_f2))


class FDS(Problem):
    r"""n_features = 10 (default), n_objectives = 3

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = \sum_i i (x_i - i)^4 / n^2, \\
        f_2(x) = \exp(\sum_i x_i / n) + \|x\|^2, \\
        f_3(x) = \sum_i i (n - i + 1) \exp(-x_i) / (n (n + 1)).
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_1(x) = 4 / n^2 \sum_i i (x_i - i)^3, \\
        \nabla f_2(x) = \exp(\sum_i x_i / n) / n + 2 x, \\
        \nabla f_3(x) = - [i (n - i + 1) \exp(-x_i) / (n (n + 1))]_i
        \end{gathered}

    Reference: Fliege, J., Graña Drummond, L.M., Svaiter, B.F.: Newton’s method for multiobjective optimization. SIAM J. Optim. 20(2), 602–626 (2009)
    """

    def __init__(
        self,
        n_features: int = 10,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=3,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )
        self.one_to_n = np.arange(self.n_features) + 1
        self.conv_n = self.one_to_n * self.one_to_n[::-1]

    def f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = np.inner(self.one_to_n, (x - self.one_to_n) ** 4) / self.n_features**2
        f2 = np.exp(x.sum() / self.n_features) + np.linalg.norm(x) ** 2
        f3 = np.inner(self.conv_n, np.exp(-x)) / (
            self.n_features * (self.n_features + 1)
        )
        return np.array([f1, f2, f3])

    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        jac_f1 = 4 / self.n_features**2 * self.one_to_n * (x - self.one_to_n) ** 3
        jac_f2 = np.exp(x.sum() / self.n_features) / self.n_features + 2 * x
        jac_f3 = -self.conv_n * np.exp(-x) / (self.n_features * (self.n_features + 1))
        return np.vstack((jac_f1, jac_f2, jac_f3))


class ZDT1(Problem):
    r"""n_features = 30 (default), n_objectives = 2

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = x_1, \\
        f_2(x) = h(x) \left( 1 - \sqrt{\frac{x_1}{h(x)}} \right),
        \end{gathered}

    where

    .. math::

        h(x) = 1 + \frac{9}{n - 1} \sum_{i=2}^n x_i.

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_1(x) = (1, 0, \dots, 0)^\top, \\
        \nabla f_2(x) = (- \frac{\sqrt{h(x) / x_1}}{2}, \frac{9}{2 (n - 1)} (1 - \sqrt{x_1 / h(x)}), \dots, \frac{9}{2 (n - 1)} (1 - \sqrt{x_1 / h(x)}) )^\top.
        \end{gathered}

    Reference: Zitzler, E., Deb, K., Thiele, L.: Comparison of multiobjective evolutionary algorithms: empirical results. Evolutionary Computation, IEEE Transactions on 8(2), 257–271 (2000)
    """

    def __init__(self, n_features: int = 30, bounds = None) -> None:
        super().__init__(n_features=n_features, n_objectives=2, bounds=bounds)

    def f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = x[0]
        h = 1 + 9 / (self.n_features - 1) * np.sum(x[1:])
        f2 = h * (1 - np.sqrt(f1 / h))
        return np.array([f1, f2])

    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        jac_f1 = np.zeros(self.n_features)
        jac_f1[0] = 1
        h = 1 + 9 / (self.n_features - 1) * np.sum(x[1:])
        jac_f2 = np.full(
            self.n_features, 9 * (2 - np.sqrt(x[0] / h)) / 2 / (self.n_features - 1)
        )
        jac_f2[0] = -np.sqrt(h / x[0]) / 2
        return np.vstack((jac_f1, jac_f2))

class ZDT4(Problem):
    r"""n_features = 10, n_objectives = 2

    Non Convex Problem
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=10,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        f1 = x[0]
        h = 1 + 10*(self.n_features - 1) + np.sum((x[1:])**2 - 10*np.cos(4*(np.pi)*(x[1:])))
        f2 = h * (1- np.sqrt(f1 / h))

        return np.array([f1, f2])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)
        jac_f1[0] = 1
        h = 1 + 10*(self.n_features - 1) + np.sum((x[1:])**2 - 10*np.cos(4*(np.pi)*(x[1:])))
        
        for i in range(1, self.n_features):
            jac_f2[i] = (2*x[i] + 40*np.pi*np.sin(4*np.pi*x[i])) * (2 - np.sqrt(x[0] / h)) / 2

        jac_f2[0] = -np.sqrt(h / x[0]) / 2
        return np.vstack((jac_f1, jac_f2))  


class TOI4(Problem):
    r"""n_features = 4, n_objectives = 2

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = x_1^2 + x_2^2 + 1,
        f_2(x) = 0.5((x_1 - x_2)^2 + (x_3 - x_4)^2) + 1.
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_1(x) = (2 x_1, 2 x_2, 0, 0)^\top,
        \nabla f_2(x) = (x_1 - x_2, x_2 - x_1, x_3 - x_4, x_4 - x_3)^\top.
        \end{gathered}

    Reference: Toint, Ph.L.: Test problems for partially separable optimization and results for the routine PSPMIN. Tech. Rep. 83/4, Department of Mathematics, University of Namur, Brussels (1983)
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=4,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = x[0] ** 2 + x[1] ** 2 + 1
        f2 = 0.5 * ((x[0] - x[1]) ** 2 + (x[2] - x[3]) ** 2) + 1
        return np.array([f1, f2])

    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        jac_f1 = np.zeros(self.n_features)
        jac_f1[0] = 2 * x[0]
        jac_f1[1] = 2 * x[1]
        jac_f2 = np.zeros(self.n_features)
        jac_f2[0] = x[0] - x[1]
        jac_f2[1] = -jac_f2[0]
        jac_f2[2] = x[2] - x[3]
        jac_f2[3] = -jac_f2[2]
        return np.vstack((jac_f1, jac_f2))


class TRIDIA(Problem):
    r"""n_features = 3, n_objectives = 3

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = (2 x_1 - 1)^2,
        f_2(x) = 2 (2 x_1 - x_2)^2,
        f_3(x) = 3 (2 x_2 - x_3)^2.
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_1(x) = (8 x_1 - 4, 0, 0)^\top,
        \nabla f_2(x) = (16 x_1 - 8 x_2, 4 x_2 - 8 x_1, 0)^\top,
        \nabla f_3(x) = (0, 24 x_2 - 12 x_3, 6 x_3 - 12 x_2)^\top.
        \end{gathered}

    Reference: Toint, Ph.L.: Test problems for partially separable optimization and results for the routine PSPMIN. Tech. Rep. 83/4, Department of Mathematics, University of Namur, Brussels (1983)
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=3,
            n_objectives=3,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        return np.array(
            [
                (2 * x[0] - 1) ** 2,
                2 * (2 * x[0] - x[1]) ** 2,
                3 * (2 * x[1] - x[2]) ** 2,
            ]
        )

    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        return np.array(
            [
                [8 * x[0] - 4, 0, 0],
                [16 * x[0] - 8 * x[1], 4 * x[1] - 8 * x[0], 0],
                [0, 24 * x[1] - 12 * x[2], 6 * x[2] - 12 * x[1]],
            ]
        )


class LinearFunctionRank1(Problem):
    r"""n_features = 10 (default), n_objectives = 4 (default)

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_i(x) = \left( i \sum_{j = 1}^n j x_j - 1 \right)^2, \quad i = 1, \dots, 4.
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_i(x) = \left[ 2 i k \left( i \sum_{j = 1}^n j x_j - 1 \right) \right]_k
        \end{gathered}

    Reference: Moré, J.J., Garbow, B.S., Hillstrom, K.E.: Testing unconstrained optimization software. ACM T. Math. Softw. 7(1), 17–41 (1981)
    """

    def __init__(
        self,
        n_features: int = 10,
        n_objectives: int = 4,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=n_objectives,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )
        self.range_n_objectives = np.arange(1, self.n_objectives + 1)
        self.range_n_features = np.arange(1, self.n_features + 1)

    def f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        return (self.range_n_objectives * np.inner(self.range_n_features, x) - 1) ** 2

    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        return (
            2
            * self.range_n_objectives[:, None]
            * self.range_n_features
            * (
                self.range_n_objectives[:, None] * np.inner(self.range_n_features, x)
                - 1
            )
        )

class KW2(Problem):
    r"""n_features = 2, n_objectives = 2

    
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=2,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = -3*((1-x[0])**2)*np.exp(-x[0]**2-(x[1]+1)**2) \
                + 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2) \
                + 3*np.exp(-(x[0]+2)**2-x[1]**2) - 0.5*(2*x[0] + x[1])

        f2 = -3*((1+x[1])**2)*np.exp(-x[1]**2-(1-x[0])**2) + \
                10*(- x[1]/5 + x[0]**5 + x[1]**3)*np.exp(-x[0]**2-x[1]**2) \
                + 3*np.exp(-(2-x[1])**2 - x[0]**2)
        
        return np.array([f1, f2])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)

        jac_f1[0] = 6*(1-x[0])*(1 + x[0] + x[0]**2)*np.exp(-x[0]**2-(x[1]+1)**2) + \
                    10*np.exp(-(x[0]**2)-(x[1]**2))*((x[0]/5-(x[0]**3) - x[1]**5)*(-2*x[0]) + 1/5 - 3*(x[0]**2)) \
                    - 6*np.exp(-(x[0]+2)**2-x[1]**2)*(x[0]+2) - 1
    
        jac_f1[1] = 6 * ((1-x[0])**2) * (x[1]+1) * np.exp(-x[0]**2-(x[1]+1)**2) \
                    + 10*np.exp(-(x[0]**2)-(x[1]**2))*((x[0]/5-(x[0]**3) - x[1]**5)*(-2*x[1]) - 5*(x[1]**4)) \
                    - 6*np.exp(-(x[0]+2)**2 - (x[1]**2) )*x[1] - 0.5
        
        jac_f2[0] = -6*((1+x[1])**2)*(1-x[0])*np.exp(-x[1]**2-(1-x[0])**2) \
                    + 10*np.exp(-x[0]**2-x[1]**2)*((-x[1]/5 + x[0]**5 + x[1]**3)*(-2*x[0]) + 5*(x[0]**4)) \
                    - 6*np.exp(-(2-x[1])**2 - x[0]**2)*x[0]
        
        jac_f2[1] = 6*(1+x[1])*(x[1]**2 + x[1] - 1) * np.exp(-x[1]**2-(1-x[0])**2) \
                    + 10*np.exp(-x[0]**2-x[1]**2)*((-x[1]/5 + x[0]**5 + x[1]**3)*(-2*x[1]) + 3*(x[1]**2) - 1/5) \
                    + 6*np.exp(-(2-x[1])**2 - x[0]**2)*(2-x[1])
        
        return np.vstack((jac_f1, jac_f2))        
    

class DD(Problem):
    r"""n_features = 5, n_objectives = 2

    Non Convex Problem
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=5,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 
        f2 = 3*x[0] + 2*x[1] - x[2]/3 + 0.01*(x[3] - x[4])**3
        
        return np.array([f1, f2])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = 2*x
        jac_f2 = np.zeros(self.n_features)

        jac_f2[0] = 3
        jac_f2[1] = 2
        jac_f2[2] = -1/3
        jac_f2[3] = 0.03*(x[3] - x[4])**2
        jac_f2[4] = -0.03*(x[3] - x[4])**2
        
        return np.vstack((jac_f1, jac_f2))        
    

class Rosenbrock(Problem):
    r"""n_features = 4, n_objectives = 3

    Non Convex Problem
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=4,
            n_objectives=3,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        f1 = 100 * ((x[1]-(x[0]**2))**2) + (x[1]-1)**2
        f2 = 100 * ((x[2]-(x[1]**2))**2) + (x[2]-1)**2
        f3 = 100 * ((x[3]-(x[2]**2))**2) + (x[2]-1)**2

        return np.array([f1, f2, f3])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)
        jac_f3 = np.zeros(self.n_features)

        jac_f1[0] = -400*x[0]*(x[1] - x[0]**2)
        jac_f1[1] = 200*(x[1] - x[0]**2) + 2*(x[1] - 1)
        jac_f2[1] = -400*x[1]*(x[2] - x[1]**2)
        jac_f2[2] = 200*(x[2] - x[1]**2) + 2*(x[2] - 1)
        jac_f3[2] = -400*x[2]*(x[3] - x[2]**2)
        jac_f3[3] = 200*(x[3] - x[2]**2) + 2*(x[3] - 1)
        
        return np.vstack((jac_f1, jac_f2, jac_f3))  
    

class MOP3(Problem):
    r"""n_features = 2, n_objectives = 2

    Non Convex Problem
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=2,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        A1 = 0.5*np.sin(1) - 2*np.cos(1) + np.sin(2) - 1.5*np.cos(2)
        A2 = 1.5*np.sin(1) - np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2)
        B1 = 0.5*np.sin(x[0]) - 2*np.cos(x[0]) + np.sin(x[1]) - 1.5*np.cos(x[1])
        B2 = 1.5*np.sin(x[0]) - np.cos(x[0]) + 2*np.sin(x[1]) - 0.5*np.cos(x[1])
        f1 = -1 - (A1 - B1)**2 - (A2 - B2)**2
        f2 = -1*(x[0]+3)**2 - (x[1]+1)**2

        return np.array([f1, f2])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)

        A1 = 0.5*np.sin(1) - 2*np.cos(1) + np.sin(2) - 1.5*np.cos(2)
        A2 = 1.5*np.sin(1) - np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2)
        B1 = 0.5*np.sin(x[0]) - 2*np.cos(x[0]) + np.sin(x[1]) - 1.5*np.cos(x[1])
        B2 = 1.5*np.sin(x[0]) - np.cos(x[0]) + 2*np.sin(x[1]) - 0.5*np.cos(x[1])
        jac_f1[0] = 2*(A1 - B1)*(0.5*np.cos(x[0]) + 2*np.sin(x[0])) + 2*(A2 - B2)*(1.5*np.cos(x[0]) + np.sin(x[0]))
        jac_f1[1] = 2*(A1 - B1)*(np.cos(x[1]) + 1.5*np.sin(x[1])) + 2*(A2 - B2)*(2*np.cos(x[1]) + 0.5*np.sin(x[1]))
        jac_f2[0] = -2*(x[0] + 3)
        jac_f2[1] = -2*(x[1] + 1)

        return np.vstack((jac_f1, jac_f2))

class MOP5(Problem):
    r"""n_features = 2, n_objectives = 3

    Non Convex Problem
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=2,
            n_objectives=3,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        f1 = 0.5*(x[0]**2 + x[1]**2) + np.sin(x[0]**2 + x[1]**2)
        f2 = (3*x[0]-2*x[1]+4)**2/8 + (x[0]-x[1]+1)**2/27 + 15
        f3 = 1/(x[0]**2 + x[1]**2 + 1) - 1.1*np.exp(-(x[0]**2 + x[1]**2))

        return np.array([f1, f2, f3])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)
        jac_f3 = np.zeros(self.n_features)

        jac_f1[0] = x[0] + 2*x[0]*np.cos(x[0]**2 + x[1]**2)
        jac_f1[1] = x[1] + 2*x[1]*np.cos(x[0]**2 + x[1]**2)
        jac_f2[0] = 3*(3*x[0] - 2*x[1] + 4)/4 + 2*(x[0] - x[1] + 1)/27
        jac_f2[1] = -1*(3*x[0] - 2*x[1] + 4)/2 - 2*(x[0] - x[1] + 1)/27
        jac_f3[0] = -2*x[0]/(x[0]**2 + x[1]**2 + 1)**2 + 2.2*x[0]*np.exp(-(x[0]**2 + x[1]**2))
        jac_f3[1] = -2*x[1]/(x[0]**2 + x[1]**2 + 1)**2 + 2.2*x[1]*np.exp(-(x[0]**2 + x[1]**2))
        
        return np.vstack((jac_f1, jac_f2, jac_f3))
    
class MOP7(Problem):
    r"""n_features = 2, n_objectives = 3

    Convex Problem
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=2,
            n_objectives=3,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        f1 = ((x[0]-2)**2) / 2 + ((x[1]+1)**2) / 13 + 3
        f2 = (x[0]+x[1]-3)**2/36 + (-x[0]+x[1]+2)**2/8 - 17
        f3 = (x[0]+2*x[1]-1)**2/175 + (2*x[1]-x[0])**2/17 - 13

        return np.array([f1, f2, f3])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)
        jac_f3 = np.zeros(self.n_features)

        jac_f1[0] = x[0] - 2
        jac_f1[1] = 2*(x[1] + 1)/13
        jac_f2[0] = (x[0] + x[1] - 3)/18 - (-x[0] + x[1] + 2)/4
        jac_f2[1] = (x[0] + x[1] - 3)/18 + (-x[0] + x[1] + 2)/4
        jac_f3[0] = 2*(x[0] + 2*x[1] - 1)/175 - 2*(2*x[1] - x[0])/17
        jac_f3[1] = 4*(x[0] + 2*x[1] - 1)/175 + 4*(2*x[1] - x[0])/17
        
        return np.vstack((jac_f1, jac_f2, jac_f3))
    
class Far1(Problem):
    r"""n_features = 2, n_objectives = 2

    Non Convex Problem
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=2,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        f1 = -2*np.exp(15*(-1*(x[0] - 0.1)**2 - x[1]**2)) \
            - np.exp(20*(-1*(x[0] - 0.6)**2 - (x[1]-0.6)**2)) \
            + np.exp(20*(-1*(x[0] + 0.6)**2 - (x[1]-0.6)**2)) \
            + np.exp(20*(-1*(x[0] - 0.6)**2 - (x[1]+0.6)**2)) \
            + np.exp(20*(-1*(x[0] + 0.6)**2 - (x[1]+0.6)**2)) 
            
        f2 = 2*np.exp(20*(-x[0]**2 - x[1]**2)) \
            + np.exp(20*(-1*(x[0] - 0.4)**2 - (x[1]-0.6)**2)) \
            - np.exp(20*(-1*(x[0] + 0.5)**2 - (x[1]-0.7)**2)) \
            - np.exp(20*(-1*(x[0] - 0.5)**2 - (x[1]+0.7)**2)) \
            + np.exp(20*(-1*(x[0] + 0.4)**2 - (x[1]+0.8)**2)) 

        return np.array([f1, f2])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)

        jac_f1[0] = - 60*(x[0] - 0.1)*np.exp(15*(-1*(x[0] - 0.1)**2 - x[1]**2)) \
                    + 40*(x[0] - 0.6)*np.exp(20*(-1*(x[0] - 0.6)**2 - (x[1]-0.6)**2)) \
                    - 40*(x[0] + 0.6)*np.exp(20*(-1*(x[0] + 0.6)**2 - (x[1]-0.6)**2)) \
                    - 40*(x[0] - 0.6)*np.exp(20*(-1*(x[0] - 0.6)**2 - (x[1]+0.6)**2)) \
                    - 40*(x[0] + 0.6)*np.exp(20*(-1*(x[0] + 0.6)**2 - (x[1]+0.6)**2))

        jac_f1[1] = - 60*x[1]      *np.exp(15*(-1*(x[0] - 0.1)**2 - x[1]**2)) \
                    + 40*(x[1]-0.6)*np.exp(20*(-1*(x[0] - 0.6)**2 - (x[1]-0.6)**2)) \
                    - 40*(x[1]-0.6)*np.exp(20*(-1*(x[0] + 0.6)**2 - (x[1]-0.6)**2)) \
                    - 40*(x[1]+0.6)*np.exp(20*(-1*(x[0] - 0.6)**2 - (x[1]+0.6)**2)) \
                    - 40*(x[1]+0.6)*np.exp(20*(-1*(x[0] + 0.6)**2 - (x[1]+0.6)**2))

        jac_f2[0] = - 80*x[0]      *np.exp(20*(-x[0]**2 - x[1]**2)) \
                    - 40*(x[0]-0.4)*np.exp(20*(-1*(x[0] - 0.4)**2 - (x[1]-0.6)**2)) \
                    + 40*(x[0]+0.5)*np.exp(20*(-1*(x[0] + 0.5)**2 - (x[1]-0.7)**2)) \
                    + 40*(x[0]-0.5)*np.exp(20*(-1*(x[0] - 0.5)**2 - (x[1]+0.7)**2)) \
                    - 40*(x[0]+0.4)*np.exp(20*(-1*(x[0] + 0.4)**2 - (x[1]+0.8)**2))

        jac_f2[1] = - 80*x[1]      *np.exp(20*(-x[0]**2 - x[1]**2)) \
                    - 40*(x[1]-0.6)*np.exp(20*(-1*(x[0] - 0.4)**2 - (x[1]-0.6)**2)) \
                    + 40*(x[1]-0.7)*np.exp(20*(-1*(x[0] + 0.5)**2 - (x[1]-0.7)**2)) \
                    + 40*(x[1]+0.7)*np.exp(20*(-1*(x[0] - 0.5)**2 - (x[1]+0.7)**2)) \
                    - 40*(x[1]+0.8)*np.exp(20*(-1*(x[0] + 0.4)**2 - (x[1]+0.8)**2))      
        
        return np.vstack((jac_f1, jac_f2))
    
class IKK1(Problem):
    r"""n_features = 2, n_objectives = 3

    Convex Problem
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=2,
            n_objectives=3,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        f1 = x[0]**2
        f2 = (x[0]-20)**2
        f3 = x[1]**2

        return np.array([f1, f2, f3])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)
        jac_f3 = np.zeros(self.n_features)

        jac_f1[0] = 2*x[0]
        jac_f2[0] = 2*(x[0]-20)
        jac_f3[1] = 2*x[1]
        
        return np.vstack((jac_f1, jac_f2, jac_f3))
    
class VFM1(Problem):
    r"""n_features = 2, n_objectives = 3

    Convex Problem
    """

    def __init__(
        self,
        l1_ratios: Optional[Sequence] = None,
        l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=2,
            n_objectives=3,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        f1 = x[0]**2 + (x[1]-1)**2
        f2 = x[0]**2 + (x[1]+1)**2 + 1
        f3 = (x[0]-1)**2 + x[1]**2 + 2

        return np.array([f1, f2, f3])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)
        jac_f3 = np.zeros(self.n_features)

        jac_f1[0] = 2*x[0]
        jac_f1[1] = 2*(x[1]-1)
        
        jac_f2[0] = 2*x[0]
        jac_f2[1] = 2*(x[1]+1)

        jac_f3[0] = 2*(x[0]-1)
        jac_f3[1] = 2*x[1]

        return np.vstack((jac_f1, jac_f2, jac_f3))
    
class DLTZ2(Problem):
    r"""n_features = 12, n_objectives = 3

    Non Convex Problem
    """

    def __init__(
        self,
        n_features = 12,
        # l1_ratios: Optional[Sequence] = None,
        # l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=3,
            # l1_ratios=l1_ratios,
            # l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        

        g = np.sum((x[2:]-0.5)**2)
        f1 = (1+g) * np.cos(x[0] * np.pi / 2) * np.cos(x[1] * np.pi / 2)
        f2 = (1+g) * np.cos(x[0] * np.pi / 2) * np.sin(x[1] * np.pi / 2)
        f3 = (1+g) * np.sin(x[0] * np.pi / 2)

        return np.array([f1, f2, f3])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)
        jac_f3 = np.zeros(self.n_features)

        g = np.sum((x[2:]-0.5)**2)

        jac_f1 = 2 * np.cos(x[0] * np.pi / 2) * np.cos(x[1] * np.pi / 2) * (x - 0.5)
        jac_f1[0] = -np.pi/2 * (1+g) * np.sin(x[0] * np.pi / 2) * np.cos(x[1] * np.pi / 2)
        jac_f1[1] = -np.pi/2 * (1+g) * np.cos(x[0] * np.pi / 2) * np.sin(x[1] * np.pi / 2)

        jac_f2 = 2 * np.cos(x[0] * np.pi / 2) * np.sin(x[1] * np.pi / 2) * (x - 0.5)
        jac_f2[0] = -np.pi/2 * (1+g) * np.sin(x[0] * np.pi / 2) * np.sin(x[1] * np.pi / 2)
        jac_f2[1] = np.pi/2 * (1+g) * np.cos(x[0] * np.pi / 2) * np.cos(x[1] * np.pi / 2)

        jac_f3 = 2 * np.sin(x[0] * np.pi / 2) * (x - 0.5)
        jac_f3[0] = np.pi/2 * (1+g) * np.cos(x[0] * np.pi / 2)
        jac_f3[1] = 0
        
        return np.vstack((jac_f1, jac_f2, jac_f3))
    
class DLTZ5(Problem):
    r"""n_features = 12, n_objectives = 3

    Non Convex Problem
    """

    def __init__(
        self,
        n_features = 12,
        n_objectives=3,
        # l1_ratios: Optional[Sequence] = None,
        # l1_shifts: Optional[Sequence] = None,
        bounds: Optional[
            Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]
        ] = None,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=n_objectives,
            # l1_ratios=l1_ratios,
            # l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        g = np.sum((x[2:]-0.5)**2)

        y = (1 + 2 * g * x[1]) / (2 + 2 * g) 
        f1 = (1+g) * np.cos(x[0] * np.pi / 2) * np.cos(y * np.pi / 2)
        f2 = (1+g) * np.cos(x[0] * np.pi / 2) * np.sin(y * np.pi / 2)
        f3 = (1+g) * np.sin(x[0] * np.pi / 2)

        return np.array([f1, f2, f3])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = np.zeros(self.n_features)
        jac_f2 = np.zeros(self.n_features)
        jac_f3 = np.zeros(self.n_features)

        
        return np.vstack((jac_f1, jac_f2, jac_f3))
    

class NonConvex_Quadratic(Problem):
    r"""n_features = 2, n_objectives = 2

    NonConvex Problem
    """

    def __init__(
        self,
        n_features = 2,
        n_objectives=2,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=n_objectives,
        )
        self.n_features = n_features
        D1 = np.random.normal(0, 1, (self.n_features, self.n_features))
        self.A1 = D1.T + D1
        D2 = np.random.normal(0, 1, (self.n_features, self.n_features))
        self.A2 = D2.T + D2
        self.b = np.random.normal(0, 1, self.n_features)

        # import pdb; pdb.set_trace()

        eig1 = np.sort(np.linalg.eigvals(self.A1))
        eig2 = np.sort(np.linalg.eigvals(self.A2))

        l1 = abs(eig1[0])
        l2 = abs(eig2[0])
        L1  = max(abs(eig1[0]), eig1[-1])
        L2  = max(abs(eig2[0]), eig2[-1])
        self.L = max(L1, L2)
        self.l = max(l1, l2)

        t = np.random.uniform(0, 1, 1)
        self.s = max(1, 10*t) 

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")

        f1 = 1/2 * (x.T @ self.A1 @ x) + self.b.T @ x
        f2 = 1/2 * (x.T @ self.A2 @ x) + self.b.T @ x

        return np.array([f1, f2])
    
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
       
        jac_f1 = self.A1 @ x + self.b
        jac_f2 = self.A2 @ x + self.b
        
        return np.vstack((jac_f1, jac_f2))
        
class SCAD(Problem):
    r"""n_features = 2, n_objectives = 2

    NonConvex Problem
    """

    def __init__(
        self,
        n_features=4,
        n_objectives=2,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=n_objectives,
        )
        self.n_features = n_features
        self.m = 2

        self.A1 = np.random.normal(0, 1, (self.m, self.n_features))
        self.A2 = np.random.normal(0, 1, (self.m, self.n_features))

        eps  = np.random.normal(0, 0.01, (self.m, 1))
        x_star = sparse.random(self.n_features, 1, 0.02)

        self.b1 = np.squeeze(self.A1 @ x_star + eps)
        self.b2 = np.squeeze(self.A2 @ x_star + eps)
        
        self.c = 3.7
        self.kappa = 0.1

        eig1 = np.sort(np.linalg.eigvals(self.A1.T @ self.A1))[-1]
        eig2 = np.sort(np.linalg.eigvals(self.A2.T @ self.A2))[-1]
        self.l = 1/(self.c-1)
        L1  = max(eig1, self.l)
        L2  = max(eig2, self.l)
        self.L = max(L1, L2)

        

    def f(self, x : np.ndarray ) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")

        f1 = 1/2 * np.linalg.norm(self.A1 @ x - self.b1)**2 - 0.5 * np.linalg.norm(x)**2 / (self.c-1)
        f2 = 1/2 * np.linalg.norm(self.A2 @ x - self.b2)**2 - 0.5 * np.linalg.norm(x)**2 / (self.c-1)

        return np.array([f1, f2])
    
    def g_x(self, x: np.ndarray) -> np.ndarray:
        
        # def g_kappa(self, theta: float) -> float:
        #     if theta <= self.kappa:
        #         return self.kappa * theta
        #     elif theta <= self.c * self.kappa:
        #         return (2 * self.kappa * self.c * theta - theta**2 - self.kappa**2) / (2 * (self.c - 1))
        #     else: 
        #         return (self.c+1)*(self.kappa**2)/2
        
        # def g_kappa(self, theta: float) -> float:
        #     return theta

        def g_kappa(self, theta: float) -> float:
            return 0
        
        sigma = 0
        for i in range(self.n_features):
            sigma += g_kappa(self, np.abs(x[i]))

        norm_x_term =  0.5 * np.linalg.norm(x)**2 / (self.c-1)

        return sigma + norm_x_term
     
    def jac_f(self, x: np.ndarray) -> np.ndarray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        
        # import pdb; pdb.set_trace()
        jac_f1 = self.A1.T @ (self.A1 @ x - self.b1) - x / (self.c - 1)
        jac_f2 = self.A2.T @ (self.A2 @ x - self.b2) - x / (self.c - 1)
        return np.vstack((jac_f1, jac_f2))