import contextlib
import inspect
import json
import os
import pickle
from logging import INFO, StreamHandler, getLogger
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import numpy as np
import pandas as pd
import scienceplots
from joblib import Parallel, delayed
from scipy.optimize import OptimizeResult
from tqdm.auto import tqdm

# np.random.seed(42)

from zfista.metrics import (
    calculate_metrics,
    extract_function_values,
    extract_non_dominated_points,
    spread_metrics,
)
from zfista.problems import (
    FDS,
    JOS1,
    SD,
    TOI4,
    TRIDIA,
    IKK1,
    VFM1,
    NonConvex_Quadratic,
    SCAD,
    MOP7,
    Problem,
    # LinearFunctionRank1,
    # KW2,
    # Rosenbrock,
    # DD,
    # ZDT4,
    # MOP3,
    # MOP5,
    # ZDT1,
    # Far1,
    # DLTZ2,
    # DLTZ5, 
    

)

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
# plt.style.use(["science", "bright"])
plt.switch_backend("agg")


@contextlib.contextmanager
def tqdm_joblib(total: Optional[int] = None, **kwargs) -> Generator:
    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()


def create_directory(problem: Problem, experiment_name: str) -> str:
    directory = os.path.join("results", experiment_name, problem.name)
    os.makedirs(directory, exist_ok=True)
    return directory


def show_Pareto_front(
    res_normal: List[OptimizeResult],
    fname: str,
    iters: int = 10,
    s: float = 15,
    alpha: float = 0.75,
    elev: float = 15,
    azim: float = 130,
    linewidth: float = 0.1,
) -> None:
    if len(res_normal[0].fun) > 3:
        return
    F_normal = extract_function_values(res_normal)
    F_0 = np.array([res.allfuns[0] for res in res_normal])

    normal_color = "#6536FF"
    initial_color = "#8e44ad"

    common_style = {"s": s, "alpha": alpha, "linewidth": linewidth}

    fig = plt.figure(figsize=(7.5, 7.5), dpi=100)
    if len(res_normal[0].fun) == 2:
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.scatter(
            F_0[:, 0],
            F_0[:, 1],
            label="Initial point",
            marker="x",
            color=initial_color,
            **common_style,
        )
        ax.scatter(
            F_normal[:, 0],
            F_normal[:, 1],
            label="Normal",
            marker="o",
            color=normal_color,
            **common_style,
        )
       

        F_iter_normal = np.array(
            [res.allfuns[iters] for res in res_normal if res.nit >= iters]
        )

        if len(F_iter_normal) > 0:
            ax.scatter(
                F_iter_normal[:, 0],
                F_iter_normal[:, 1],
                label=f"Normal ({iters} iters)",
                marker="o",
                edgecolors=normal_color,
                facecolors="none",
                **common_style,
            )

        ax.set_xlabel(f"$F_1$")
        ax.set_ylabel(f"$F_2$")
        ax.legend()
    elif len(res_normal[0].fun) == 3:
        ax = fig.add_subplot(111, projection="3d")
        # ax = fig.gca(projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.scatter(
            F_0[:, 0],
            F_0[:, 1],
            F_0[:, 2],
            label="Initial point",
            marker="x",
            color=initial_color,
            **common_style,
        )
        ax.scatter(
            F_normal[:, 0],
            F_normal[:, 1],
            F_normal[:, 2],
            label="Normal",
            marker="o",
            color=normal_color,
            **common_style,
        )

        ax.set_xlabel(f"$F_1$")
        ax.set_ylabel(f"$F_2$")
        ax.set_zlabel(f"$F_3$")
        ax.legend()

    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def show_Pareto_front_feasible(
    result_normal: List[OptimizeResult],
    fname: str,
    iters: int = 1,
    s: float = 15,
    elev: float = 15,
    azim: float = 130,
    problem = None,
    problem_params = None,
    linspace_pts = 18,
    num_samples = 15
    ) -> None:

    n_features = problem.n_features
    low, high = problem_params.get("low"), problem_params.get("high")
    if isinstance(low, (int, float)):
        low = np.full(n_features, low)
    if isinstance(high, (int, float)):
        high = np.full(n_features, high)
    x0 = [np.linspace(low[i], high[i], linspace_pts if len(result_normal[0].fun) == 2 else linspace_pts) for i in range(n_features)]
    
    x0_grid = np.meshgrid(*x0)
    x0_grid_flatten = np.vstack([x0_grid[i].flatten() for i in range(n_features)]).T
    F_feasible = np.array(list(set(map(tuple, [problem.f(x0) + problem.g(x0) for x0 in x0_grid_flatten]))))
    
    res_normal = np.random.choice(result_normal, num_samples)
    F_normal = extract_function_values(res_normal)
    F_0 = np.array([res.allfuns[0] for res in res_normal])

    initial_common_style =  {"color": "#00FFF8" , "s": s, "alpha": 1, "linewidth": 0.75}
    normal_common_style =   {"s": s, "alpha": 1, "linewidth": 0.4, "facecolor": "#2EFF00", "edgecolor": "#000000" }
    iters_common_style =    {"color": "#000000", "s": 4, "alpha": 1 }
    line_common_style =     {"color": "#EC00FF", "linewidth": 0.7}
    feasible_common_style = {"color": "#DDDDDD", "s": 10, "alpha": 1 }

    fig = plt.figure(figsize=(7.5, 7.5), dpi=100)
    plt.rcParams.update({'font.size': 15})

    if len(res_normal[0].fun) == 2:
        ax = fig.add_subplot(111)

        ##Plotting feasible region
        ax.scatter(
            F_feasible[:, 0],
            F_feasible[:, 1],

            marker="o",
            **feasible_common_style,
        )

        for res in res_normal:
            funs = np.array(res.allfuns)
            ax.scatter(
                funs[1:, 0],
                funs[1:, 1],
                marker="o",
                **iters_common_style,
            )

            plt.plot(
                funs[:, 0],
                funs[:, 1],
                **line_common_style,
                )
        
        ##Plotting pareto front

        ax.scatter(
            F_0[:, 0],
            F_0[:, 1],
            # label="Initial point",
            marker="o",
            **initial_common_style,
        )
        ax.scatter(
            F_normal[:, 0],
            F_normal[:, 1],
            # label="Normal",
            marker="o",
            **normal_common_style,
        )

        ax.set_xlabel(f"$F_1$")
        ax.set_ylabel(f"$F_2$")
        # ax.legend()


    elif len(res_normal[0].fun) == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)


        ## Plotting feasible region for 3D, currently not working so commented it.
        # ax.plot_surface(
        #     F_feasible[:, 2].reshape(-1, 1),
        #     F_feasible[:, 0].reshape(-1, 1),
        #     F_feasible[:, 1].reshape(-1, 1), 
        #     color='gray', 
        #     alpha=0.8)
        
        ax.scatter(
            F_feasible[:, 2],
            F_feasible[:, 0],
            F_feasible[:, 1],
            
            marker="o",
            **feasible_common_style,
        )

        for res in res_normal:
            funs = np.array(res.allfuns)
            ax.scatter(
                funs[1:, 2],
                funs[1:, 0],
                funs[1:, 1],
                
                marker="o",
                **iters_common_style,
            )

            plt.plot(
                funs[:, 2],
                funs[:, 0],
                funs[:, 1],
                
                **line_common_style,
                )
        
        ##Plotting pareto front
        
        ax.scatter(
            F_0[:, 2],
            F_0[:, 0],
            F_0[:, 1],
            # label="Initial point",
            marker="o",
            **initial_common_style,
        )
        ax.scatter(
            F_normal[:, 2],
            F_normal[:, 0],
            F_normal[:, 1],
            
            # label="Normal",
            marker="o",
            **normal_common_style,
        )

        ax.set_xlabel(f"$F_1$")
        ax.set_ylabel(f"$F_2$")
        ax.set_zlabel(f"$F_3$")
        # ax.legend()
    
    
    if problem.problem_name in ["NonConvex_Quadratic", "SCAD"]:
        plt.title(f"{problem.problem_name}")
    else:
        plt.title(f"{problem.problem_name} with {problem.g_name}")
    
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    

def show_error_decay(
    res_normal: OptimizeResult,
    fname: str,
):
    normal_color = "#6536FF"

    plt.figure(figsize=(7.5, 7.5), dpi=100)
    plt.yscale("log")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\|x^k - y^k\|_\infty$")
    plt.plot(res_normal.allerrs, label="Normal", color=normal_color, linestyle="dashed")
    plt.legend()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def save_results(
    problem: Problem,
    experiment_name: str,
    res_normal: List[OptimizeResult],
    metrics: Dict[str, Dict[str, float]],
    problem_params: Dict[str, float],
) -> None:
    logger.info("Saving results...")
    directory = create_directory(problem, experiment_name)
    with open(os.path.join(directory, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    show_Pareto_front(
        res_normal,
        os.path.join(directory, "pareto_front.pdf"),
    )
    show_Pareto_front_feasible(
        res_normal,
        os.path.join(directory, "pareto_front_feasible.pdf"),
        problem = problem,
        problem_params=problem_params
    )
    show_error_decay(
        res_normal[0],
        os.path.join(directory, "error_decay.pdf"),
    )
    logger.info("Results saved.")


def load_or_run_results(
    file_name: str,
    directory: str,
    overwrite: bool,
    run_fn: Callable,
) -> List[OptimizeResult]:
    if not overwrite and os.path.exists(os.path.join(directory, file_name)):
        try:
            logger.info(f"Loading {file_name}...")
            with open(os.path.join(directory, file_name), "rb") as f:
                results = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {file_name} due to: {e}")
            results = run_fn()
            with open(os.path.join(directory, file_name), "wb") as f:
                pickle.dump(results, f)
    else:
        logger.info(f"Running {file_name}...")
        results = run_fn()
        with open(os.path.join(directory, file_name), "wb") as f:
            pickle.dump(results, f)
    return results


def benchmark(
    problem: Problem,
    experiment_name: str,
    low: Union[float, np.ndarray],
    high: Union[float, np.ndarray],
    nesterov: bool = False,
    inertial: bool = False,
    inertial_params: Tuple[float, float] = (0, 0),
    n_samples: int = 100,
    overwrite: bool = False,
    max_iter: int = 100000000,
    tol_internal: float = 1e-11,
    verbose: bool = False,
) -> Tuple[List[OptimizeResult], List[OptimizeResult], List[OptimizeResult]]:
    directory = create_directory(problem, experiment_name)

    initial_points = np.random.uniform(
        low=low, high=high, size=(n_samples, problem.n_features)
    )

    with tqdm_joblib(total=n_samples, desc="Normal") as progress_bar:
        res_normal = load_or_run_results(
            "normal_results.pkl",
            directory,
            overwrite,
            lambda: Parallel(n_jobs=-1)
                (delayed(problem.minimize_proximal_gradient)(
                    x0,
                    return_all=True,
                    max_iter=max_iter,
                    tol_internal=tol_internal,
                    verbose=verbose,
                    nesterov=nesterov,
                    inertial=inertial,
                    inertial_params=inertial_params
                )
                for x0 in initial_points
            )
        )

    return res_normal


def generate_performance_profiles(
    performance_ratios: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    performance_profiles: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for ratio_key, algorithm_ratios in performance_ratios.items():
        performance_profiles[ratio_key] = {}
        for algorithm, ratios in algorithm_ratios.items():
            thresholds = []
            percentages = []
            for i, ratio in enumerate(sorted(ratios)):
                thresholds.append(ratio)
                percentages.append((i + 1) / len(ratios))
            performance_profiles[ratio_key][algorithm] = (
                np.array(thresholds),
                np.array(percentages),
            )
    return performance_profiles


def plot_performance_profiles(
    metric_key: str,
    algorithm_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
    fname: str,
) -> None:
    plt.figure(figsize=(7.5, 7.5), dpi=100)
    plt.xlabel("Threshold")
    plt.ylabel("Percentage of Problems")
    for algorithm, profile in algorithm_profiles.items():
        thresholds, percentages = profile
        plt.step(thresholds, percentages, label=algorithm)
    plt.legend()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def main(overwrite=False, verbose=False) -> None:
    problem_classes = [
        JOS1,
        SD,
        TOI4,
        FDS,
        IKK1,
        TRIDIA,
        VFM1,
        MOP7,
        NonConvex_Quadratic,
        SCAD,
        # LinearFunctionRank1,
        # ZDT1,
        # ZDT4,
        # Rosenbrock,
        # DD,
        # ZDT4,
        # MOP3,
        # MOP5,
        # Far1,
        # KW2,
        # DLTZ2,
        # DLTZ5, 
        
    ]

    bounds_list = {
        JOS1: (0, np.inf),
        SD: (1e-6, np.inf),
        TRIDIA: (0, np.inf),
        TOI4: (0, np.inf),
        VFM1: (0, np.inf),
        IKK1: (0, np.inf),
        
        # ZDT1: (1e-6, np.inf),
        # ZDT4: (1e-6, np.inf),
        # DD: (0, np.inf),
        # MOP3: (1e-6, np.inf),
        # ## MOP5 and MOP7 is not compitible with bounds
        # Far1: (0, np.inf),
        # KW2: (0, np.inf),
        # DLTZ2: (0, np.inf),
        # DLTZ5: (0, np.inf),
    }

    n_samples = 50
    ## set inertial to True if running GIPGM, else accelearated(Tanabe et.al.) will run automatically
    inertial = True 
    nesterov = False if inertial else True
    beta = 0
    if inertial == True:
        beta = 0
    alpha = 2*beta/(1+beta)

    problems = []

    for problem_class in problem_classes:
        constructor_params = inspect.signature(problem_class.__init__).parameters  # type: ignore
        problem = problem_class()
        problems.append(problem)
        if "l1_ratios" in constructor_params and "l1_shifts" in constructor_params:
            n_features = problem.n_features
            n_objectives = problem.n_objectives
            l1_ratios = (np.arange(n_objectives) + 1) / n_features
            l1_shifts = np.arange(n_objectives)
            problems.append(problem_class(l1_ratios=l1_ratios, l1_shifts=l1_shifts))
        if (problem_class in bounds_list) and "bounds" in constructor_params:
                problems.append(problem_class(bounds=bounds_list[problem_class]))

    if inertial: 
        experiment_name = f"proximal_vs_inertial_proximal_samples{n_samples}_beta{beta}"
    elif nesterov:
        experiment_name = f"proximal_vs_accelerated_proximal"
    print(experiment_name)
    # zdt4_low, zdt4_high = np.full(10, -5), np.full(10, 5)
    # zdt4_low[0], zdt4_high[0] = 1/100, 1

    problem_parameters = {
        "JOS1": {"low": -2, "high": 4},
        "FDS": {"low": -2, "high": 2},
        "SD": {"low": [1, np.sqrt(2), np.sqrt(2), 1], "high": [3, 3, 3, 3]},
        "ZDT1": {"low": 0, "high": 0.01},
        "TOI4": {"low": -2, "high": 5},
        "TRIDIA": {"low": -1, "high": 1},
        "KW2": {"low": -3, "high": 3},
        "LinearFunctionRank1": {"low": -1, "high": 1},
        "Rosenbrock": {"low": -2, "high": 2},
        "DD": {"low": -20, "high": 20},
        # "ZDT4": {"low": zdt4_low, "high": zdt4_high},
        "MOP3": {"low": -np.pi, "high": np.pi},
        "MOP5": {"low": -30, "high": 30},
        "MOP7": {"low": -1, "high": 1},
        "Far1": {"low": -1, "high": 1},
        "IKK1": {"low": -50, "high": 50},
        "VFM1": {"low": -2, "high": 2},
        "Far": {"low": -1, "high": 1},
        "KW2": {"low": -3, "high": 3},
        "DLTZ2": {"low": 0, "high": 1},
        "DLTZ5": {"low": 0, "high": 1},
        "NonConvex_Quadratic": {"low": -1, "high": 1},
        "SCAD": {"low": -1, "high": 1},
    }
    
    # performance_ratios: Dict[str, Dict[str, List[float]]] = {}
    df_rows = []

    for problem in problems:
        logger.info(f"Running benchmark for {problem.name}...")
        print(type(problem).__name__, "Problem Name")
        problem_params = problem_parameters.get(type(problem).__name__)
        low, high = problem_params.get("low"), problem_params.get("high")  # type: ignore
        
        if inertial == True:
            if problem.problem_name == "NonConvex_Quadratic":
                L, l = problem.L, problem.l
                beta = 0.5 # 0.98*(L/(L+l))
                alpha = 2*beta/(1+beta)

            if problem.problem_name == "SCAD":
                L, l = problem.L, problem.l
                beta = 0.5 # 0.7 * (L / (L + l))
                alpha = 2 * beta / (1 + beta)

        inertial_params = (alpha, beta)
        res_normal = benchmark(
            problem,
            experiment_name,
            low,
            high,
            nesterov,
            inertial,
            inertial_params,
            n_samples,
            overwrite=overwrite,
            verbose=verbose,
        )
        
        '''While evaluating, if loading the true pareto front from some other expt, 
        ## put the name of the expt in create directory function's 2nd parameter.
        Default = None, Calculates the true pareto front from same expt.
        '''
        if nesterov:
            directory_true_front = None
        else:
            directory_true_front = None
            # directory_true_front = create_directory(problem, f"proximal_vs_inertial_proximal_samples{n_samples}_beta0")
            
        metrics, ratios = calculate_metrics(
            ("Normal", res_normal), 
            directory = directory_true_front
        )

        save_results(
            problem, experiment_name, res_normal, metrics, problem_params
        )
        logger.info(f"Benchmark completed for {problem.name}.")


        # # Add metrics to dataframe
        # for metric_key, algorithms_metrics in metrics.items():
        #     df_rows.extend(
        #         [
        #             {
        #                 "problem": problem.name,
        #                 "algorithm": algorithm,
        #                 "metric": metric_key,
        #                 "value": metric,
        #             }
        #             for algorithm, metric in algorithms_metrics.items()
        #         ]
        #     )

        # for ratio_key, algorithms_ratios in ratios.items():
        #     if ratio_key not in performance_ratios:
        #         performance_ratios[ratio_key] = {}
        #     for algorithm, ratio in algorithms_ratios.items():
        #         if algorithm not in performance_ratios[ratio_key]:
        #             performance_ratios[ratio_key][algorithm] = []
        #         performance_ratios[ratio_key][algorithm].append(ratio)
        

    # performance_profiles = generate_performance_profiles(performance_ratios)
    # for ratio_key, algorithm_profiles in performance_profiles.items():
    #     logger.info(f"Plotting performance profile for {ratio_key}...")
    #     plot_performance_profiles(
    #         ratio_key,
    #         algorithm_profiles,
    #         os.path.join("results", experiment_name, f"{ratio_key}.pdf"),
    #     )
    # # Save metrics to csv
    # df = pd.concat([pd.DataFrame(row, index=[0]) for row in df_rows], ignore_index=True)
    # df.columns = ["problem", "algorithm", "metric", "value"]
    # df.to_csv(os.path.join("results", experiment_name, "metrics.csv"), index=False)
