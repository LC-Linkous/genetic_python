#! /usr/bin/python3

##--------------------------------------------------------------------\
#   genetic_python
#   './genetic_python/src/main_test_graph.py'
#   Test/example driver with VISUALIZATION for the genetic_algorithm class.
#   Mirrors the optimizer set main_test_graph.py: the optimizer stays headless, 
#   ALL rendering lives here in the driver via draw_current_best(), using
#   the SAME shape_decoder the headless objective uses (so on-screen and
#   scored images can't drift). matplotlib is used for the display so the
#   demo has no wxPython dependency; swap the body of draw_current_best()
#   for a wx canvas if integrating into the GUI.
#
#   Author(s): Lauren Linkous
#   Last update: June 6, 2026
##--------------------------------------------------------------------\

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from genetic_algorithm import GeneticAlgorithm
import obj_func_img.configs_F as func_configs
from obj_func_img.shape_decoder import decode_genes


def render_candidate(active_genes, num_features, shape_type, w, h):
    # pure render used by the visualizer; identical decode path to func_F
    img = Image.new('RGB', (w, h), (255, 255, 255))
    drw = ImageDraw.Draw(img)
    for s in decode_genes(active_genes, num_features, shape_type, w, h):
        x, y, sw, sh, rgb = s['x'], s['y'], s['w'], s['h'], s['rgb']
        if s['type'] == 'circle':
            r = sw
            drw.ellipse([x - r, y - r, x + r, y + r], fill=rgb)
        else:
            drw.rectangle([x, y, x + sw, y + sh], fill=rgb)
    return np.asarray(img)


class TestGraph:
    def __init__(self, ref_path=None):
        self.ctr = 0
        self.W = self.H = 120

        # reference image: load if given, else a synthetic stand-in
        if ref_path:
            ref = Image.open(ref_path).convert('RGB').resize((self.W, self.H))
        else:
            ref = Image.new('RGB', (self.W, self.H), (250, 250, 250))
            d = ImageDraw.Draw(ref)
            d.ellipse([20, 20, 80, 95], fill=(40, 80, 200))
            d.rectangle([60, 50, 110, 105], fill=(220, 120, 40))
        self.ref_arr = np.asarray(ref)

        self.shape_type = func_configs.SHAPE_TYPE
        self.num_features = func_configs.NUM_FEATURES
        func_configs.SET_REFERENCE(self.ref_arr, self.num_features, self.shape_type,
                                   canvas_w=self.W, canvas_h=self.H)

        # optimizer config
        POPULATION_SIZE = 25
        MAX_GENES = 500
        TOL = 10 ** -9
        MAXIT = 10000
        LB, UB = func_configs.LB, func_configs.UB
        TARGETS = func_configs.TARGETS

        self.best_eval = 1e9
        self.suppress_output = True
        self.allow_update = True

        opt_params = {
            'POPULATION_SIZE': [POPULATION_SIZE],
            'NUM_GENES': [8], 'MAX_GENES': [MAX_GENES],
            'NUM_FEATURES': [self.num_features],
            'MUTATION_RATE': [0.05], 'SCALE_FACTOR': [0.25],
            'SPAWN_CHANCE': [0.20], 'REMOVE_CHANCE': [0.03],
            'MUTATE_ATTEMPTS': [1], 'CROSS_RATE': [0.7],
        }
        opt_df = pd.DataFrame(opt_params)

        self.myOptimizer = GeneticAlgorithm(LB, UB, TARGETS, TOL, MAXIT,
                                             func_configs.OBJECTIVE_FUNC,
                                             func_configs.CONSTR_FUNC,
                                             opt_df, parent=None,
                                             evaluate_threshold=False,
                                             obj_threshold=np.zeros_like(TARGETS),
                                             decimal_limit=5)

        # matplotlib: reference | current best
        self.fig, (self.axL, self.axR) = plt.subplots(1, 2, figsize=(8, 4))
        self.axL.set_title("Reference"); self.axL.imshow(self.ref_arr); self.axL.axis('off')
        self.axR.axis('off')

    def debug_message_printout(self, txt):
        if txt is None:
            return
        print("[" + time.strftime("%H:%M:%S", time.localtime()) + "] " + str(txt))

    def record_params(self):
        pass

    # THE VISUALIZER -- lives in the driver, not the optimizer
    def draw_current_best(self):
        genes = self.myOptimizer.get_optimized_genes()
        if genes is None or len(genes) < 1:
            return
        img = render_candidate(genes, self.num_features, self.shape_type, self.W, self.H)
        self.axR.clear(); self.axR.axis('off')
        self.axR.set_title("Best, iter " + str(self.ctr) +
                           " (genes " + str(len(genes)) + ")")
        self.axR.imshow(img)
        plt.pause(0.001)
        self.ctr += 1

    def run(self, draw_every=200):
        while not self.myOptimizer.complete():
            self.myOptimizer.step(self.suppress_output)
            self.myOptimizer.call_objective(self.allow_update)
            iter, eval = self.myOptimizer.get_convergence_data()
            if eval < self.best_eval:
                self.best_eval = eval
            if iter % draw_every == 0:
                self.draw_current_best()
        self.draw_current_best()
        print("Optimized Outputs", self.myOptimizer.get_optimized_outs())
        print("Optimization ended.")


if __name__ == "__main__":
    ref_path = "src/obj_func_img/images/starrynight.jpg"
    ga = TestGraph(ref_path)
    ga.run()
