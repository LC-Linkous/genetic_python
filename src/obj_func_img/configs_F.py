#! /usr/bin/python3

##--------------------------------------------------------------------\
#   genetic_python
#   './src/obj_func_img/configs_F.py'
#   Constant values + problem binding for the image-drawing objective.
# 
#  Formatted for automating objective function integration to work with 
#  the objective function set. 
#
#   NOTE on bounds: LB/UB describe ONE GENE's feature ranges (length =
#   NUM_FEATURES), all in [0,1]. The optimizer tiles these across MAX_GENES
#   to build the fixed-width decision vector.
#
#   Author(s): Lauren Linkous
#   Last update: June 6, 2026
##--------------------------------------------------------------------\

import sys
try:  # for outside func calls
    sys.path.insert(0, './genetic_python/src/')
    from obj_func_img.func_F import func_F, set_reference
    from obj_func_img.constr_F import constr_F
except Exception:  # for local
    from func_F import func_F, set_reference
    from constr_F import constr_F

OBJECTIVE_FUNC = func_F
CONSTR_FUNC = constr_F
SET_REFERENCE = set_reference  # driver calls this to bind the reference image
OBJECTIVE_FUNC_NAME = "obj_func_img.func_F"
CONSTR_FUNC_NAME = "obj_func_img.constr_F"

# ---- shape selection (drives NUM_FEATURES) ----
# 0 = circle (6 feats), 1 = square (5 feats), 2 = rectangle (7 feats)
SHAPE_TYPE = 1
_FEATS_BY_SHAPE = {0: 6, 1: 5, 2: 7}
NUM_FEATURES = _FEATS_BY_SHAPE[SHAPE_TYPE]

# per-gene feature bounds, all normalized [0,1]
LB = [[0.0] * NUM_FEATURES]
UB = [[1.0] * NUM_FEATURES]

IN_VARS = NUM_FEATURES      # per-gene decision width
OUT_VARS = 1                # single objective: image error
TARGETS = [0]               # minimize error toward 0
GLOBAL_MIN = None
