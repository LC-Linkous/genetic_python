##--------------------------------------------------------------------\
#   GA_Drawing_Example
#   'population.py'
#   Class for a population of organisms. Uses 1 organism with X genes to draw.
#   Contains functions for saving check points
#   modified from: https://cosmiccoding.com.au/tutorials/genetic_part_one
#
#   Author: Lauren Linkous (LINKOUSLC@vcu.edu)
#   November 19, 2022
##--------------------------------------------------------------------\

# python lib imports
from colour import Color
from numpy.random import random
import numpy as np

# project file imports
from organism import Organism
import constants as c

class Population:
    def __init__(self, parent, ref):
        self.parent = parent
        self.ref = ref
        self.w, self.h, *d = self.ref.shape

        #evolution comparison params
        self.currentBestOrganism = None

        #user params
        self.imgUpdate = 250 # default = 250
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
            drawArr = []
            # scale x and y and add
            draw_x = int(x * self.w)  # scale x for drawing
            drawArr.append(draw_x)
            draw_y = int(y * self.h)  # scale y for drawing
            drawArr.append(draw_y)

            # update the gene list we can work with to not include the vals above
            remaining_gene = gene[2:]  # remove x and y
            remaining_gene = remaining_gene[:-3]  # remove hsl

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

            self.parent.gui.addShape(drawArr)  # draw shape on parent GUI
            ctr = ctr + 1  # increment for report
        self.currentGenes = ctr     # set counter
        self.parent.gui.onPaint()   # trigger paint event

    def spawn(self, numGenes=1, numFeatures=3):
        # create a population of individual organisms with
        # user controlled number of genes and features
        #  random(3, 4) creates 3 genes with 4 features
        # numGenes number of genes, with a complexity of

        # in this case only a single individual is created
        random_genes = random((numGenes, numFeatures))
        print(random_genes)
        self.currentBestOrganism = Organism(random_genes)
        self.calcFitness(self.currentBestOrganism)

    def calcFitness(self, org):
        # fitness of new organism is calculated by drawing it and then
        # comparing to the reference (original) image.

        #update the drawing
        self.drawOrganism(org)
        image = self.parent.gui.getCurrentImage()
        #get the current image
        im = image.GetDataBuffer()
        arr = np.frombuffer(im, dtype='uint8', count=-1, offset=0)
        # convert to array
        image = np.reshape(arr, (c.PANEL_HEIGHT, c.PANEL_WIDTH, 3))
        diff = image - self.ref
        org.fitness = -np.mean(np.abs(diff)) - 1e-5 * org.chromosome.size
        org.visual = image

    def step(self, time, mutationRate=0.01, scaleFactor=0.1, spawnChance=0.3, removeChance=0.3):
        # each step works by working like 'generations'.
        # the organism is mutated (or not) based on the user input parameters
        # If the new organism is better than the last one, we update to the new one

        o = self.currentBestOrganism.mutate(mutationRate=mutationRate, scaleFactor=scaleFactor, spawnChance=spawnChance, removeChance=removeChance)
        self.calcFitness(o)
        if o.fitness > self.currentBestOrganism.fitness:
            self.currentBestOrganism = o

        #create a check point
        # + append to the summary file to keep track of what parameters are in use
        # NOTE: it is possible to change parameters and load from last check point,
        # but parameters are not always changing
        if time % self.imgUpdate == 0:
            checkpt = time/self.imgUpdate
            savePath = self.outDir + f"{time // self.imgUpdate:04d}.png"
            #save progress
            np.save(self.outDir + self.saveFile, self.currentBestOrganism.chromosome)
            self.parent.gui.saveImage(savePath)
            print("Check point # ", checkpt, " Current number of genes: ", self.currentGenes)
            with open(self.outDir + self.saveSummary, "a", encoding="utf-8") as f:
                f.write("time: " + str(time) +
                        "\tmutation: " + str(mutationRate) +
                        "\tscale: " + str(scaleFactor) +
                        "\tspawn: " + str(spawnChance) +
                        "\tremove: " + str(removeChance) +
                        "\tgenes: " + str(self.currentGenes) + "\n")