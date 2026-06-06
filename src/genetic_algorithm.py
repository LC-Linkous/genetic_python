#! /usr/bin/python3

##--------------------------------------------------------------------\
#   genetic_python
#   './src/genetic_algorithm.py'
#   Genetic algorithm class. Restructured from the GA_Drawing_Example
#       (organism/population classes) to match the state-machine optimizer
#       structure used by particle_swarm.py in the AntennaCAT optimizer suite.
#
#       Key structural choices (see project notes):
#       - ONE individual evaluated per step() call, exactly like the swarm.
#         A full generation is bred when current_individual wraps to 0;
#         offspring are staged and then doled out one objective call at a time.
#       - Fixed-width decision vectors with a per-gene ACTIVE MASK so that
#         organisms can still gain/lose genes (spawn/remove) while M stays a
#         strict-dimension float array that round-trips through export/import.
#       - The objective function is the ONLY place fitness is measured. The GA
#         never renders, never imports a GUI. Image-diff, equation, or
#         simulation objectives all satisfy obj_func(X, NO_OF_OUTS)->(F, noErr).
#
#   Author(s): Lauren Linkous, Jonathan Lundquist
#   Last update: June 6, 2026
##--------------------------------------------------------------------\

import numpy as np
from numpy.random import Generator, MT19937
import sys
np.seterr(all='raise')


class GeneticAlgorithm:
    # arguments should take the form:
    # genetic_algorithm([[float,...]], [[float,...]], [[float,...]], float, int,
    #                   func, func,
    #                   dataFrame,
    #                   class obj,
    #                   bool, [int,...],
    #                   int)
    #
    # opt_df contains class-specific tuning parameters:
    # POPULATION_SIZE : int    number of individuals (this is the agent count)
    # NUM_GENES       : int    active genes an individual starts with
    # MAX_GENES       : int    hard ceiling on genes (sets the fixed M width)
    # NUM_FEATURES    : int    values per gene (decision sub-vector per gene)
    # MUTATION_RATE   : float  chance/scale driver for per-feature mutation
    # SCALE_FACTOR    : float  std-dev tuning for mutation + gene blending
    # SPAWN_CHANCE    : float  chance a mutation adds/removes a gene
    # REMOVE_CHANCE   : float  chance that a spawn event is a removal
    # MUTATE_ATTEMPTS : int    mutate-and-pick retries (population2 behavior)
    # CROSS_RATE      : float  prob a shared-index gene comes from parent A
    #
    def __init__(self, lbound, ubound, targets, E_TOL, maxit,
                 obj_func, constr_func,
                 opt_df,
                 parent=None,
                 evaluate_threshold=False, obj_threshold=None,
                 decimal_limit=4):

        # Optional parent class for message passing / constraint logging
        self.parent = parent

        self.number_decimals = int(decimal_limit)

        # evaluation method for targets
        # True : evaluate as thresholds based on obj_threshold
        # False: evaluate as true targets
        if evaluate_threshold == False:
            self.evaluate_threshold = False
            self.obj_threshold = None
        else:
            if not (len(obj_threshold) == len(targets)):
                self.debug_message_printout("WARNING: THRESHOLD option selected. \
                Dimensions for THRESHOLD do not match TARGET array. Defaulting to TARGET search.")
                self.evaluate_threshold = False
                self.obj_threshold = None
            else:
                self.evaluate_threshold = evaluate_threshold
                self.obj_threshold = np.array(obj_threshold).reshape(-1, 1)

        # unpack the opt_df standardized vals
        NO_OF_INDIVIDUALS = int(opt_df['POPULATION_SIZE'][0])
        self.num_genes_start = int(opt_df['NUM_GENES'][0])
        self.max_genes = int(opt_df['MAX_GENES'][0])
        self.num_features = int(opt_df['NUM_FEATURES'][0])
        self.mutationRate = float(opt_df['MUTATION_RATE'][0])
        self.scaleFactor = float(opt_df['SCALE_FACTOR'][0])
        self.spawnChance = float(opt_df['SPAWN_CHANCE'][0])
        self.removeChance = float(opt_df['REMOVE_CHANCE'][0])
        self.mutate_attempts = int(opt_df['MUTATE_ATTEMPTS'][0])
        self.cross_rate = float(opt_df['CROSS_RATE'][0])

        # variable- vs fixed-length problems.
        # True  : genes may spawn/remove; organism complexity evolves (drawing, etc.)
        # False : every individual is exactly NUM_GENES active genes, no spawn/remove.
        #         This is the mode for tuning a fixed set of numerical parameters.
        # Defaults to True so existing opt_df configs behave exactly as before.
        if 'VARIABLE_LENGTH' in opt_df.columns:
            self.variable_length = bool(opt_df['VARIABLE_LENGTH'][0])
        else:
            self.variable_length = True

        # in fixed-length mode, MAX_GENES is forced to NUM_GENES so the decision
        # vector width equals the active width and no slots sit dormant.
        if self.variable_length == False:
            self.max_genes = self.num_genes_start

        # bounds check (same shape rules as the swarm)
        heightl = np.shape(lbound)[0]
        widthl = np.shape(lbound)[1]
        heightu = np.shape(ubound)[0]
        widthu = np.shape(ubound)[1]

        lbound = np.array(lbound[0])
        ubound = np.array(ubound[0])

        self.rng = Generator(MT19937())

        if ((heightl > 1) and (widthl > 1)) \
           or ((heightu > 1) and (widthu > 1)) \
           or (heightu != heightl) \
           or (widthl != widthu):
            if self.parent == None:
                pass
            else:
                self.parent.debug_message_printout("Error lbound and ubound must be 1xN-dimensional \
                                                        arrays  with the same length")
        else:
            # lbound/ubound describe ONE GENE's feature bounds (length = num_features).
            # The full decision vector for an individual is max_genes copies of this,
            # flattened, so M stays a strict fixed-width float array.
            self.gene_lbound = lbound
            self.gene_ubound = ubound

            # full flat bounds across all gene slots
            self.lbound = np.tile(lbound, self.max_genes)
            self.ubound = np.tile(ubound, self.max_genes)

            self.vector_len = self.max_genes * self.num_features

            '''
            self.M            : (pop x vector_len) flat decision vectors (genes flattened)
            self.GeneMask     : (pop x max_genes) 1.0 = active gene, 0.0 = dormant
            self.Active       : (pop,) individual-level activity (suite parity)
            self.Gb           : best decision vector found (1 x vector_len)
            self.F_Gb         : fitness (error) at the global best (1 x output_size)
            self.GbMask       : (1 x max_genes) gene mask of the global best
            self.Pb           : per-individual surviving decision vector (parent)
            self.F_Pb         : fitness of each surviving parent
            self.PbMask       : (pop x max_genes) gene mask of each surviving parent
            self.targets      : target output values
            self.maxit        : max iterations (objective calls)
            self.E_TOL        : convergence tolerance on ||F_Gb||
            self.obj_func     : objective. (X, NO_OF_OUTS) -> (F, noError)
            self.constr_func  : constraint. (X) -> bool
            self.iter         : objective-call count
            self.current_individual : index of agent being evaluated this step
            self.number_of_individuals : population size
            self.allow_update : gate for whether objective call updates state
            self.Flist        : last evaluated Flist (suite parity)
            self.Fvals        : last raw objective output (suite parity)
            self.Staged       : (pop x vector_len) offspring waiting to be scored
            self.StagedMask   : (pop x max_genes) offspring gene masks
            self.F_Staged     : (pop x output_size) offspring fitness, filled per step
            '''

            self.output_size = len(targets)
            self.number_of_individuals = NO_OF_INDIVIDUALS

            # ---- spawn the initial population ----
            self.M = np.zeros((NO_OF_INDIVIDUALS, self.vector_len))
            self.GeneMask = np.zeros((NO_OF_INDIVIDUALS, self.max_genes))
            for ind in range(NO_OF_INDIVIDUALS):
                vec, mask = self._random_individual()
                self.M[ind] = vec
                self.GeneMask[ind] = mask

            self.Active = np.ones((NO_OF_INDIVIDUALS))
            self.Gb = sys.maxsize * np.ones((1, self.vector_len))
            self.F_Gb = sys.maxsize * np.ones((1, self.output_size))
            self.GbMask = np.ones((1, self.max_genes))
            self.Pb = 1.0 * self.M
            self.F_Pb = sys.maxsize * np.ones((NO_OF_INDIVIDUALS, self.output_size))
            self.PbMask = 1.0 * self.GeneMask

            self.targets = np.array(targets).reshape(-1, 1)
            self.maxit = maxit
            self.E_TOL = E_TOL
            self.obj_func = obj_func
            self.constr_func = constr_func
            self.iter = 0
            self.current_individual = 0
            self.allow_update = 0
            self.Flist = []
            self.Fvals = []

            # staging buffers for the "breed on wrap" generation
            self.Staged = 1.0 * self.M
            self.StagedMask = 1.0 * self.GeneMask
            self.F_Staged = sys.maxsize * np.ones((NO_OF_INDIVIDUALS, self.output_size))

            # the very first generation's individuals ARE the staged candidates,
            # so the first pass simply evaluates the random spawn.
            self.generation = 0
            self.Mlast = 1.0 * self.ubound

            self.debug_message_printout("genetic algorithm successfully initialized")

    # ----------------------------------------------------------------\
    # gene/individual construction helpers
    # ----------------------------------------------------------------\
    def _random_gene(self):
        # one gene = num_features values sampled within the per-gene bounds
        variation = self.gene_ubound - self.gene_lbound
        gene = np.multiply(self.rng.random((self.num_features,)), variation) + self.gene_lbound
        return np.round(gene, self.number_decimals)

    def _random_individual(self):
        # returns (flat vector length vector_len, mask length max_genes)
        vec = np.zeros(self.vector_len)
        mask = np.zeros(self.max_genes)
        n_active = min(self.num_genes_start, self.max_genes)
        for g in range(self.max_genes):
            gene = self._random_gene()
            vec[g * self.num_features:(g + 1) * self.num_features] = gene
            if g < n_active:
                mask[g] = 1.0
        return vec, mask

    def _genes_view(self, vec):
        # reshape a flat decision vector into (max_genes x num_features)
        return np.array(vec).reshape(self.max_genes, self.num_features)

    def _flatten(self, genes):
        return np.array(genes).reshape(-1)

    # ----------------------------------------------------------------\
    # genetic operators (ported 1:1 from organism.py / population2.py,
    # rewritten to act on fixed-width vector+mask instead of variable arrays)
    # ----------------------------------------------------------------\
    def _mutate(self, vec, mask):
        # mirrors Organism.mutate: either a spawn/remove event, or a
        # feature-level mutation whose intensity decays as the organism grows.
        genes = self._genes_view(vec).copy()
        mask = mask.copy()
        active_idx = np.where(mask > 0.5)[0]
        n_active = len(active_idx)

        randomNum = self.rng.random()
        if self.variable_length and (randomNum < self.spawnChance):
            if (randomNum < self.removeChance * self.spawnChance) and (n_active > 1):
                # remove a random active gene -> flip its mask off
                drop = active_idx[self.rng.integers(0, n_active)]
                mask[drop] = 0.0
            else:
                # add a gene by blending two active "parent" genes into a free slot
                free = np.where(mask < 0.5)[0]
                if len(free) > 0 and n_active >= 2:
                    a, b = active_idx[self.rng.choice(n_active, 2, replace=False)]
                    newg = 0.5 * (genes[a, :] + genes[b, :])
                    newg = newg + self.scaleFactor * self.rng.normal(size=self.num_features)
                    if self.num_features > 2:
                        newg[2] *= 0.2  # original dampened the 3rd feature on spawn
                    slot = free[0]
                    genes[slot, :] = newg
                    mask[slot] = 1.0
        else:
            # feature-level mutation, decaying step size (matches original tuning)
            size = max(n_active * self.num_features, 1)
            num_mutations = 1 + int(self.mutationRate * size)
            updateRate = self.scaleFactor / (1 + int(self.mutationRate * size)) + 2e-6
            for _ in range(num_mutations):
                g = active_idx[self.rng.integers(0, n_active)] if n_active > 0 else 0
                f = self.rng.integers(0, self.num_features)
                genes[g, f] += self.rng.normal() * updateRate

        vec_out = np.round(self._flatten(genes), self.number_decimals)
        # keep every slot within bounds so M stays valid even for dormant genes
        vec_out = np.clip(vec_out, self.lbound, self.ubound)
        return vec_out, mask

    def _crossover(self, vecA, maskA, vecB, maskB):
        # mirrors Population2.mixGenes: per shared gene index, take from A with
        # prob cross_rate, else B; trailing genes handled by which parent has them.
        genesA = self._genes_view(vecA)
        genesB = self._genes_view(vecB)
        child = np.zeros((self.max_genes, self.num_features))
        child_mask = np.zeros(self.max_genes)

        write = 0
        for g in range(self.max_genes):
            aOn = maskA[g] > 0.5
            bOn = maskB[g] > 0.5
            if aOn and bOn:
                src = genesA[g] if self.rng.random() < self.cross_rate else genesB[g]
                child[write] = src
                child_mask[write] = 1.0
                write += 1
            elif aOn:
                child[write] = genesA[g]
                child_mask[write] = 1.0
                write += 1
            elif bOn:
                if self.rng.random() < (1.0 - self.cross_rate):
                    child[write] = genesB[g]
                    child_mask[write] = 1.0
                    write += 1
        if write == 0:
            # degenerate guard: keep at least one gene
            child[0] = genesA[0]
            child_mask[0] = 1.0
        return self._flatten(child), child_mask

    def _selection_weights(self):
        # rank-based weighting identical in spirit to population2.step:
        # fitter (lower error -> we sort ascending) parents are favored.
        n = self.number_of_individuals
        weights = 1.0 - np.linspace(0, 0.2, n)
        return weights / weights.sum()

    # ----------------------------------------------------------------\
    # breeding: build the next generation into the staging buffers.
    # called when current_individual wraps to 0 (one full generation done).
    # ----------------------------------------------------------------\
    def _breed_generation(self):
        # order parents by fitness (ascending error). Pb holds the survivors.
        order = np.argsort(np.linalg.norm(self.F_Pb, axis=1))
        weights = self._selection_weights()

        for slot in range(self.number_of_individuals):
            ia, ib = order[self.rng.choice(self.number_of_individuals, 2,
                                           replace=True, p=weights)]
            childVec, childMask = self._crossover(self.Pb[ia], self.PbMask[ia],
                                                  self.Pb[ib], self.PbMask[ib])
            childVec, childMask = self._mutate(childVec, childMask)
            self.Staged[slot] = childVec
            self.StagedMask[slot] = childMask

        self.F_Staged = sys.maxsize * np.ones((self.number_of_individuals, self.output_size))
        self.generation += 1

    # ----------------------------------------------------------------\
    # objective handling  (identical contract to swarm.call_objective)
    # ----------------------------------------------------------------\
    def call_objective(self, allow_update):
        if self.Active[self.current_individual]:
            # the candidate being scored this step is the staged offspring
            X = self.Staged[self.current_individual]
            mask = self.StagedMask[self.current_individual]
            # hand the objective only the ACTIVE genes, flattened. The objective
            # (image diff / equation / simulation) decides what to do with them.
            active_genes = self._genes_view(X)[mask > 0.5].reshape(-1)
            newFVals, noError = self.obj_func(active_genes, self.output_size)
            if noError == True:
                self.Fvals = np.array(newFVals).reshape(-1, 1)
                if allow_update:
                    self.Flist = self.objective_function_evaluation(self.Fvals, self.targets)
                    self.F_Staged[self.current_individual] = np.squeeze(self.Flist)
                    self.iter = self.iter + 1
                    self.allow_update = 1
                else:
                    self.allow_update = 0
            else:
                # failed/invalid evaluation (e.g. solver crash, bad parse):
                # deactivate just like invisible_bound so it can't poison selection
                self.F_Staged[self.current_individual] = sys.maxsize * np.ones(self.output_size)
            return noError
        return True

    def objective_function_evaluation(self, Fvals, targets):
        # identical target/threshold logic to the swarm, so convergence
        # comparisons across the suite stay apples-to-apples.
        epsilon = np.finfo(float).eps
        Flist = np.zeros(len(Fvals))

        if self.evaluate_threshold == True:
            ctr = 0
            for i in targets:
                o_thres = int(self.obj_threshold[ctr].item())  # explicit scalar (NumPy 2 safe)
                t = targets[ctr].item()                        # column-vector cell -> scalar
                fv = Fvals[ctr].item()
                if o_thres == 0:
                    Flist[ctr] = abs(t - fv)
                elif o_thres == 1:
                    Flist[ctr] = epsilon if fv <= t else abs(t - fv)
                elif o_thres == 2:
                    Flist[ctr] = epsilon if fv >= t else abs(t - fv)
                else:
                    self.debug_message_printout("ERROR: unrecognized threshold value. Evaluating as TARGET")
                    Flist[ctr] = abs(t - fv)
                ctr = ctr + 1
        else:
            Flist = abs(targets - Fvals)

        return Flist

    # ----------------------------------------------------------------\
    # selection bookkeeping  (the GA analog of check_global_local)
    # ----------------------------------------------------------------\
    def _update_bests(self, individual):
        # child replaces parent only if fitter (population.step semantics),
        # and the global best tracks best-seen (for the GUI / get_optimized_*).
        childF = self.F_Staged[individual]
        if np.linalg.norm(childF) < np.linalg.norm(self.F_Pb[individual]):
            self.F_Pb[individual] = childF
            self.Pb[individual] = self.Staged[individual]
            self.PbMask[individual] = self.StagedMask[individual]

        if np.linalg.norm(self.F_Pb[individual]) < np.linalg.norm(self.F_Gb):
            self.F_Gb = np.array([self.F_Pb[individual]])
            self.Gb = np.array(self.Pb[individual])
            self.GbMask = np.array([self.PbMask[individual]])

    # ----------------------------------------------------------------\
    # convergence / completion  (identical to swarm)
    # ----------------------------------------------------------------\
    def converged(self):
        return np.linalg.norm(self.F_Gb) < self.E_TOL

    def maxed(self):
        return self.iter >= self.maxit

    def complete(self):
        return self.converged() or self.maxed()

    # ----------------------------------------------------------------\
    # the state-machine step: advance ONE individual per call
    # ----------------------------------------------------------------\
    def step(self, suppress_output):
        if not suppress_output:
            msg = "\n-----------------------------\n" + \
                "STEP #" + str(self.iter) + "\n" + \
                "-----------------------------\n" + \
                "Generation:\n" + str(self.generation) + "\n" + \
                "Current Individual:\n" + str(self.current_individual) + "\n" + \
                "Active genes:\n" + str(int(np.sum(self.StagedMask[self.current_individual]))) + "\n" + \
                "Best error (norm F_Gb):\n" + str(np.linalg.norm(self.F_Gb)) + "\n" + \
                "-----------------------------"
            self.debug_message_printout(msg)

        if self.allow_update:
            if self.Active[self.current_individual]:
                # the candidate for this individual was scored in the prior
                # call_objective; fold it into the bests now.
                self._update_bests(self.current_individual)

            self.current_individual = self.current_individual + 1

            if self.current_individual == self.number_of_individuals:
                # one full generation evaluated -> breed the next one
                self.current_individual = 0
                self.Active = np.ones((self.number_of_individuals))
                self._breed_generation()

            if self.complete() and not suppress_output:
                msg = "\nOPTIMIZATION COMPLETE:\nPoints: \n" + str(self.Gb) + "\n" + \
                    "Iterations: \n" + str(self.iter) + "\n" + \
                    "Flist: \n" + str(self.F_Gb) + "\n" + \
                    "Norm Flist: \n" + str(np.linalg.norm(self.F_Gb)) + "\n"
                self.debug_message_printout(msg)

    # ----------------------------------------------------------------\
    # export / import  (same dict-of-lists shape as swarm)
    # ----------------------------------------------------------------\
    def export_swarm(self):
        swarm_export = {
            'evaluate_threshold': [self.evaluate_threshold],
            'obj_threshold': [self.obj_threshold],
            'targets': [self.targets],
            'lbound': [self.lbound],
            'ubound': [self.ubound],
            'gene_lbound': [self.gene_lbound],
            'gene_ubound': [self.gene_ubound],
            'output_size': [self.output_size],
            'maxit': [self.maxit],
            'E_TOL': [self.E_TOL],
            'iter': [self.iter],
            'current_individual': [self.current_individual],
            'allow_update': [self.allow_update],
            # GA-specific tuning
            'max_genes': [self.max_genes],
            'num_features': [self.num_features],
            'num_genes_start': [self.num_genes_start],
            'mutationRate': [self.mutationRate],
            'scaleFactor': [self.scaleFactor],
            'spawnChance': [self.spawnChance],
            'removeChance': [self.removeChance],
            'mutate_attempts': [self.mutate_attempts],
            'cross_rate': [self.cross_rate],
            'variable_length': [self.variable_length],
            'generation': [self.generation],
            # shared / state vars
            'M': [self.M],
            'GeneMask': [self.GeneMask],
            'Active': [self.Active],
            'Gb': [self.Gb],
            'F_Gb': [self.F_Gb],
            'GbMask': [self.GbMask],
            'Pb': [self.Pb],
            'F_Pb': [self.F_Pb],
            'PbMask': [self.PbMask],
            'Flist': [self.Flist],
            'Fvals': [self.Fvals],
            'Staged': [self.Staged],
            'StagedMask': [self.StagedMask],
            'F_Staged': [self.F_Staged],
            'Mlast': [self.Mlast]
        }
        return swarm_export

    def import_swarm(self, swarm_export):
        self.evaluate_threshold = bool(swarm_export['evaluate_threshold'][0])
        self.obj_threshold = np.array(swarm_export['obj_threshold'][0])
        self.targets = np.array(swarm_export['targets'][0]).reshape(-1, 1)
        self.lbound = np.array(swarm_export['lbound'][0])
        self.ubound = np.array(swarm_export['ubound'][0])
        self.gene_lbound = np.array(swarm_export['gene_lbound'][0])
        self.gene_ubound = np.array(swarm_export['gene_ubound'][0])
        self.output_size = int(swarm_export['output_size'][0])
        self.maxit = int(swarm_export['maxit'][0])
        self.E_TOL = float(swarm_export['E_TOL'][0])
        self.iter = int(swarm_export['iter'][0])
        self.current_individual = int(swarm_export['current_individual'][0])
        self.allow_update = int(swarm_export['allow_update'][0])

        self.max_genes = int(swarm_export['max_genes'][0])
        self.num_features = int(swarm_export['num_features'][0])
        self.num_genes_start = int(swarm_export['num_genes_start'][0])
        self.mutationRate = float(swarm_export['mutationRate'][0])
        self.scaleFactor = float(swarm_export['scaleFactor'][0])
        self.spawnChance = float(swarm_export['spawnChance'][0])
        self.removeChance = float(swarm_export['removeChance'][0])
        self.mutate_attempts = int(swarm_export['mutate_attempts'][0])
        self.cross_rate = float(swarm_export['cross_rate'][0])
        if 'variable_length' in swarm_export:
            self.variable_length = bool(swarm_export['variable_length'][0])
        else:
            self.variable_length = True
        self.generation = int(swarm_export['generation'][0])

        self.M = np.array(swarm_export['M'][0])
        self.GeneMask = np.array(swarm_export['GeneMask'][0])
        self.Active = np.array(swarm_export['Active'][0])
        self.Gb = np.array(swarm_export['Gb'][0])
        self.F_Gb = np.array(swarm_export['F_Gb'][0])
        self.GbMask = np.array(swarm_export['GbMask'][0])
        self.Pb = np.array(swarm_export['Pb'][0])
        self.F_Pb = np.array(swarm_export['F_Pb'][0])
        self.PbMask = np.array(swarm_export['PbMask'][0])
        self.Flist = np.array(swarm_export['Flist'][0])
        self.Fvals = np.array(swarm_export['Fvals'][0])
        self.Staged = np.array(swarm_export['Staged'][0])
        self.StagedMask = np.array(swarm_export['StagedMask'][0])
        self.F_Staged = np.array(swarm_export['F_Staged'][0])
        self.Mlast = np.array(swarm_export['Mlast'][0])

        self.number_of_individuals = np.shape(self.M)[0]
        self.vector_len = np.shape(self.M)[1]

    # ----------------------------------------------------------------\
    # accessors  (same names/semantics as swarm)
    # ----------------------------------------------------------------\
    def get_obj_inputs(self):
        # the active genes of the individual currently being evaluated,
        # flattened -- this is what the objective/simulation receives.
        X = self.Staged[self.current_individual]
        mask = self.StagedMask[self.current_individual]
        return np.vstack(self._genes_view(X)[mask > 0.5].reshape(-1))

    def get_convergence_data(self):
        best_eval = np.linalg.norm(self.F_Gb)
        iteration = 1 * self.iter
        return iteration, best_eval

    def get_optimized_soln(self):
        # return ONLY the active genes of the global best, flattened+column
        genes = self._genes_view(self.Gb)[self.GbMask[0] > 0.5].reshape(-1)
        return genes.reshape(-1, 1)

    def get_optimized_outs(self):
        return self.F_Gb[0]

    def get_optimized_genes(self):
        # convenience for the renderer: best individual as (n_active x num_features)
        return self._genes_view(self.Gb)[self.GbMask[0] > 0.5]

    def absolute_mean_deviation_of_particles(self):
        mean_data = np.array(np.mean(self.M, axis=0)).reshape(1, -1)
        abs_data = np.zeros(np.shape(self.M))
        for i in range(0, self.number_of_individuals):
            abs_data[i] = np.squeeze(np.abs(self.M[i] - mean_data))
        return np.linalg.norm(np.mean(abs_data, axis=0))

    def debug_message_printout(self, msg):
        if self.parent == None:
            print(msg)
        else:
            self.parent.debug_message_printout(msg)