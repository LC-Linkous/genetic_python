#! /usr/bin/python3

##--------------------------------------------------------------------\
#   genetic_python
#   './genetic_python/src/main_test.py'
#   Test/example driver for the 'GeneticAlgorithm' class. 
# 
#   Test function/example for using the 'GeneticAlgorithm' class in 
#    genetic_algorithm.py. Format updates are for integration in 
#     the AntennaCAT GUI. This example is fully headless (no GUI import).
#
#   Switch OBJECTIVE FUNCTION SELECTION between the equation problems and
#   the image problem -- the optimizer code does not change at all.
#
#   Author(s): Lauren Linkous
#   Last update: June 6, 2026
##--------------------------------------------------------------------\

import pandas as pd
import numpy as np
from genetic_algorithm import GeneticAlgorithm

# OBJECTIVE FUNCTION SELECTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
#import himmelblau.configs_F as func_configs         # single objective, 2D input
#import lundquist_3_var.configs_F as func_configs     # multi objective function, fixed 1-gene
import obj_func_img.configs_F as func_configs        # image: single-objective, variable genes


if __name__ == "__main__":
    # Constant variables
    POPULATION_SIZE = 30
    TOL = 10 ** -6
    MAXIT = 10000
    MUTATE_ATTEMPTS = 1

    # Objective function dependent variables
    func_F = func_configs.OBJECTIVE_FUNC
    constr_F = func_configs.CONSTR_FUNC
    LB = func_configs.LB
    UB = func_configs.UB
    OUT_VARS = func_configs.OUT_VARS
    TARGETS = func_configs.TARGETS
    NUM_FEATURES = func_configs.IN_VARS

    # Gene configuration.
    # DEFAULT (equation/numerical objectives like himmelblau, lundquist_3_var,
    # one_dim_x_test): a fixed-length individual that IS the decision vector --
    # one gene holding all IN_VARS variables, no spawn/remove. This is the
    # correct mode for tuning a fixed set of numerical parameters.
    NUM_GENES = 1           # active genes an individual starts with
    MAX_GENES = 1           # hard ceiling (sets fixed decision-vector width)
    VARIABLE_LENGTH = False

    # --- if running the IMAGE objective, bind a reference image first ---
    if hasattr(func_configs, "SET_REFERENCE"):
        # build a simple synthetic reference so the demo runs without assets
        from PIL import Image, ImageDraw
        W = H = 120
        ref = Image.new('RGB', (W, H), (250, 250, 250))
        d = ImageDraw.Draw(ref)
        d.ellipse([20, 20, 80, 90], fill=(40, 80, 200))
        d.rectangle([60, 50, 110, 100], fill=(220, 120, 40))
        ref_arr = np.asarray(ref)
        func_configs.SET_REFERENCE(ref_arr, NUM_FEATURES, func_configs.SHAPE_TYPE,
                                   canvas_w=W, canvas_h=H)
        # the image problem is the variable-length case: many genes (shapes),
        # gene count free to evolve via spawn/remove.
        NUM_GENES = 8
        MAX_GENES = 30
        VARIABLE_LENGTH = True

    THRESHOLD = np.zeros_like(TARGETS)

    best_eval = float('inf')
    parent = None
    evaluate_threshold = False
    suppress_output = True
    allow_update = True

    opt_params = {
        'POPULATION_SIZE': [POPULATION_SIZE],
        'NUM_GENES': [NUM_GENES],
        'MAX_GENES': [MAX_GENES],
        'NUM_FEATURES': [NUM_FEATURES],
        'MUTATION_RATE': [0.03],
        'SCALE_FACTOR': [0.25],
        'SPAWN_CHANCE': [0.20],
        'REMOVE_CHANCE': [0.03],
        'MUTATE_ATTEMPTS': [MUTATE_ATTEMPTS],
        'CROSS_RATE': [0.7],
        'VARIABLE_LENGTH': [VARIABLE_LENGTH],
    }
    opt_df = pd.DataFrame(opt_params)

    myOptimizer = GeneticAlgorithm(LB, UB, TARGETS, TOL, MAXIT,
                                    func_F, constr_F,
                                    opt_df,
                                    parent=parent,
                                    evaluate_threshold=evaluate_threshold,
                                    obj_threshold=THRESHOLD,
                                    decimal_limit=5)

    while not myOptimizer.complete():
        myOptimizer.step(suppress_output)
        myOptimizer.call_objective(allow_update)
        iter, eval = myOptimizer.get_convergence_data()
        if (eval < best_eval) and (eval != 0):
            best_eval = eval
        if suppress_output:
            if iter % 100 == 0:
                print("Iteration", iter, " Best Eval", round(best_eval, 6))

    print("Optimized Solution")
    print(myOptimizer.get_optimized_soln())
    print("Optimized Outputs")
    print(myOptimizer.get_optimized_outs())