#! /usr/bin/python3

##--------------------------------------------------------------------\
#   genetic_python
#   './src/obj_func_img/constr_F.py'
#   Constraint function for the image-drawing problem. Same contract as
#   the equation constr_F: returns True if the candidate passes.
#   Features are already bounded to [0,1] by the optimizer, so the default
#   here simply confirms validity; tighten as needed per problem.
#
#   Author(s): Lauren Linkous
#   Last update: June 6, 2026
##--------------------------------------------------------------------\

import numpy as np


def constr_F(X):
    F = True
    try:
        x = np.array(X, dtype=float)
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            F = False
    except Exception:
        F = False
    return F
