##--------------------------------------------------------------------\
#   GA_Drawing_Example
#   'population.py'
#   Class for a population of organisms. Uses 1 organism with X genes to draw.
#   Contains functions for saving check points
#   modified from: https://cosmiccoding.com.au/tutorials/genetic_part_two
#
#   Author: Lauren Linkous (LINKOUSLC@vcu.edu)
#   November 12, 2022
##--------------------------------------------------------------------\

# python lib imports
from colour import Color
import numpy as np
from numpy.random import choice, random, normal
import json

# project file imports
from organism import Organism
import constants as c

class Population2:
    def __init__(self, parent, ref):
        self.parent = parent
        self.ref = ref
        self.w, self.h, *d = self.ref.shape

        self.population = []

        # user params
        self.imgUpdate = 250
        self.outDir = ""
        self.saveFile = ""
        self.saveSummary = ""
        self.square_length = 0.1

        # update info
        self.currentGenes = 0

    def setUserParams(self, imgUpdate=250, outdir="", saveFile="save.npy", summaryFile="organismSummary.txt"):
        self.imgUpdate = imgUpdate      # how often to save the images
        self.outDir = outdir            # directory to save everything
        self.saveFile = saveFile        # the npy file
        self.saveSummary = summaryFile  # summary of check point changes

    def drawOrganism(self, org, scale=0.3, minSize=0.01):
        # the scale and minSize are optional,
        # and should be set at START of process, or risk disrupting the process

        # draw the organism (the collection of genes) on the gui canvas
        # for each gene, the x & y are pulled and scaled,

        ctr = 0
        for gene in org.chromosome:
            # split known values from gene
            # the first 2 values are always x and y
            x = gene[0]
            y = gene[1]
            # the last 3 values are always hsl version of color
            hsl = gene[-3:]

            # what's going to be passed as args to the draw method
            drawArr=[]
            # scale x and y and add
            draw_x = int(x * self.w)  # scale x for drawing
            drawArr.append(draw_x)
            draw_y = int(y * self.h)  # scale y for drawing
            drawArr.append(draw_y)

            # update the gene list we can work with to not include the vals above
            remaining_gene = gene[2:] #remove x and y
            remaining_gene = remaining_gene[:-3] # remove hsl

            # the rest of the values may or may not exist based on shape
            # because coords are always (x,y) when drawing, we can use that to decide how to scale

            if len(remaining_gene) == 1:
                #probably a 'size' val. i.e. radius of circle, length of side of polygon
                size = remaining_gene[0]
                draw_size = int((size * scale + minSize) * self.w)  # scale size
                drawArr.append(draw_size)
            elif len(remaining_gene) == 0: # square.
                # squares are drawn using static sizes for this example
                l = int((self.square_length * scale + minSize) * self.w)  # scale size
                drawArr.append(l)
                drawArr.append(l)
            else:
                #shape that takes a height, width argument
                width = remaining_gene[0]
                height = remaining_gene[1]
                width = int((width * scale + minSize) * self.w)  # scale size
                height = int((height * scale + minSize) * self.w)  # scale size
                drawArr.append(width)
                drawArr.append(height)

            # convert color vals from hsl to rgb (last 3 values of gene)
            # ALWAYS the last value in the argument passed to the gui
            c = tuple(map(lambda x: int(255 * x),  Color(hsl=hsl).rgb))
            drawArr.append(c)

            self.parent.gui.addShape(drawArr) #draw shape on parent GUI
            ctr = ctr + 1           # increment for report
        self.currentGenes = ctr     # set counter
        self.parent.gui.onPaint()   # trigger paint event

    def spawn(self, populationSize=30, numGenes=10, numFeatures=6):
        # create a population of individual organisms with
        # user controlled number of genes and features
        # random(3, 4) creates 3 genes with 4 features
        # numGenes number of genes, with a complexity of
        for i in range(populationSize):
            organism = Organism(random((numGenes, numFeatures)))
            self.population.append(organism)
            self.calcFitness(organism)
        self.population = sorted(self.population, key=lambda x: -x.fitness)

    def calcFitness(self, org):
        # fitness of new organism is calculated by drawing it and then
        # comparing to the reference (original) image.

        # update the drawing
        self.drawOrganism(org)
        image = self.parent.gui.getCurrentImage()
        # get the current image
        im = image.GetDataBuffer()
        arr = np.frombuffer(im, dtype='uint8', count=-1, offset=0)
        # convert to array
        image = np.reshape(arr, (c.PANEL_HEIGHT, c.PANEL_WIDTH, 3))
        diff = image - self.ref
        org.fitness = -np.mean(np.abs(diff)) - 1e-5 * org.chromosome.size
        org.visual = image

    def mixGenes(self, a, b):
        # genes are combined from two parent genes, A and B.
        # if one chromosome is longer, a limit is place to only mix up to the length of the
        # shorter chromosome.
        # that is, if organism A has 7 genes, and B has 5, then only 5 genes of the chromosome can be mixed
        # in this example, most of the genes are kept from the first parent (note the random < 0.7)

        new_genes = []
        n_a, n_b = a.chromosome.shape[0], b.chromosome.shape[0]

        # the for loop iterates through based on longest length chromosome
        # but if the index is longer than what the organism has, genes are not mixed

        for i in range(max(n_a, n_b)):

            if i < n_a and i < n_b: # parents both have a gene
                if random() < 0.7: # 70% chance the child gets gene from A
                    new_genes.append(a.chromosome[i, :])
                else: # 30% chance the child gets gene from B
                    new_genes.append(b.chromosome[i, :])
            elif i < n_a:   # parent A only has this gene
                new_genes.append(a.chromosome[i, :])
            else: # parent B only has this gene
                if random() < 0.3: # 30% chance new child gets this gene
                    new_genes.append(b.chromosome[i, :])
            chromosome = np.array(new_genes)
        o = Organism(chromosome)
        self.calcFitness(o)
        return o

    def saveCurrentOrganisms(self, path):
        # saves a JSON file with all of the current organism data so that the simulation can be resumed
        # or backed up for later analysis
        out = [o.chromosome.tolist() for o in self.population]
        with open(path, "w") as f:
            json.dump(out, f)

    def loadOrganismData(self, path):
        # loads a JSON file of previous organism data.
        # There are no checks on if this matches current simulation settings
        with open(path) as f:
            inp = json.load(f)
        self.population = [Organism(np.array(x)) for x in inp]
        for o in self.population:
            self.calcFitness(o)

    def mutateAndPick(self, organism, mutationRate, scaleFactor, spawnChance, removeChance, attempts=10):
        # over X number of attempts, mutate and choose the best options
        # doing this over a number of attempts increases the chances
        # that you'll get a better/relevant mutation
        for i in range(attempts):
            o = organism.mutate(mutationRate=mutationRate, scaleFactor=scaleFactor, spawnChance=spawnChance, removeChance=removeChance)
            self.calcFitness(o)
            if o.fitness > organism.fitness:
                #returns early if there's a better mutation
                return o
        return organism

    def step(self, time, mutationRate=0.01, scaleFactor=0.1, spawnChance=0.3, removeChance=0.3):
        # each step works by working like 'generations'.
        # the organism is mutated (or not) based on the user input parameters
        # If the new organism is better than the last one, we update to the new one

        # Get some children by picking the fitter parents
        new_orgs = []
        weights = 1 - np.linspace(0, 0.2, len(self.population))
        for i in range(len(self.population)):
            #make a random choice
            a, b = choice(self.population, 2, replace=True, p=weights / weights.sum())
            child = self.mixGenes(a, b)
            new_orgs.append(self.mutateAndPick(child, mutationRate, scaleFactor, spawnChance, removeChance))

        # Calculate fitness,sort fitness, update population
        for o in new_orgs:
            self.calcFitness(o)
        sorted_orgs = sorted(new_orgs, key=lambda x: -x.fitness)
        self.population = sorted_orgs[:len(self.population)]


        #create a check point
        # + append to the summary file to keep track of what parameters are in use
        # NOTE: it is possible to change parameters and load from last check point,
        # but parameters are not always changing
        if time % self.imgUpdate == 0:
            checkpt = time/self.imgUpdate
            savePath = self.outDir + f"{time // self.imgUpdate:04d}.png"
            # save progress
            self.saveCurrentOrganisms(self.outDir + self.saveFile + "save.json")
            self.parent.gui.saveImage(savePath)
            print("Check point # ", checkpt, " Current number of genes: ", self.currentGenes)
            with open(self.outDir + self.saveSummary, "a", encoding="utf-8") as f:
                f.write("time: " + str(time) +
                        "\tmutation: " + str(mutationRate) +
                        "\tscale: " + str(scaleFactor) +
                        "\tspawn: " + str(spawnChance) +
                        "\tremove: " + str(removeChance) +
                        "\tgenes: " + str(self.currentGenes) + "\n")