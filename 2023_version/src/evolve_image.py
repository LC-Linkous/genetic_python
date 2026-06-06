##--------------------------------------------------------------------\
#   GA_Drawing_Example
#   'evolve_image.py'
#   Main class for driving genetic algorithm.
#   modified from: https://cosmiccoding.com.au/tutorials/genetic_part_one
#
#   Author: Lauren Linkous (LINKOUSLC@vcu.edu)
#   October 30, 2022
##--------------------------------------------------------------------\

# python lib imports
import numpy as np
import os
import wx

# project file imports
from population import Population   # for single organism
from population2 import Population2 # for multiple organisms
from organism import Organism       # organism base-class

class EvolveImage:
    def __init__(self, refImage):

        #population
        self.pop = None

        #evolution
        self.i = 0

        #GUI vars
        self.gui = None
        self.imageBitmap = None
        self.ref = refImage #the reference image

        # user set vars
        ## population settings
        self.singleOrganism = True
        self.numOrganisms = 1
        self.numGenes = 10
        self.numFeatures = 3
        ## evolution factors
        self.mutationRate = None
        self.scaleFactor = None
        self.spawnChance = None
        self.removeChance = None
        ## simulation settings
        self.steps = None
        self.imgUpdate = None
        self.outputDir = None
        self.start = 0
        #saving/saved params
        self.save = None
        self.summary = None

    def addGUI(self, gui):
         self.gui = gui

    def setEvolutionParams(self, mutationRate=0.01, scaleFactor=0.25,
                           spawnChance=0.01, removeChance=0.01):
        # user set vals
        # mutationRate = chance of mutation in an organism
        # scaleFactor =  the tuning factor for how much mutation propagates in evolution
        # spawnChance = chances for a gene to be added or removed
        # removeChance = changes that (if spawning) a gene is removed

        self.mutationRate = mutationRate
        self.scaleFactor = scaleFactor
        self.spawnChance = spawnChance
        self.removeChance = removeChance
        print("evolution parameters set")


    def setPopulationParams(self, singleOrg=True, numOrganisms=1):
        self.singleOrganism = singleOrg
        if self.singleOrganism == True:
            self.numOrganisms = 1
        else:
            self.numOrganisms = numOrganisms

        print("population parameters set")

    def setOrganismParams(self, numGenes=10, organismShape=0):
        self.numGenes = numGenes
        self.organismShape = organismShape
        if self.organismShape == 0: # circle (original tutorial shape)
            # x loc, y loc, radius, (r, g, b)
            self.numFeatures = 6
        elif self.organismShape == 1: # square
            # x loc, y loc, (r, g, b)
            # size is a set value for this example
            #self.numFeatures = 5
            # X loc, Y loc, fill
            self.numFeatures = 5
        elif self.organismShape == 2: # rectangle
            # x loc, y loc, width, height, (r, g, b)
            self.numFeatures = 7
        else:
            # circle
            print("ERROR: unrecognized organism shape option selected. defaulting to circle")
            self.organismShape = 0
            self.numFeatures = 6

        self.gui.setShapeType(self.organismShape)

        print("organism parameters set")

    def createPopulation(self):
        if self.singleOrganism == True:
            self.pop = Population(self, self.ref)
        else:
            self.pop = Population2(self, self.ref)
        self.pop.setUserParams(imgUpdate=self.imgUpdate, outdir=self.outputDir,
                               saveFile=self.save, summaryFile=self.summary)
        print("population using ", str(self.numOrganisms), " organisms created")

    def setSimulationParams(self, steps=500000, imgUpdate=250,
                            outputDir="output/", saveFile="save", summaryFile="organismSummary.txt"):
        self.steps = steps
        self.imgUpdate = imgUpdate
        self.outputDir = outputDir
        if saveFile == "save":
            if self.singleOrganism == True:
                self.save = saveFile + ".npy"
            else:
                self.save = saveFile + ".json"
        else:
            self.save = saveFile
        self.summary = summaryFile
        # if output directory doesnt exist, then create it:
        # create directories recursively:
        os.makedirs(self.outputDir, exist_ok=True)
        print("simulation parameters set")

    def printParameterSummary(self):
        print("---------------------------SUMMARY---------------------------")
        print("population details:")
        print("\tsingle organism: ", self.singleOrganism, "\t num organisms: ", self.numOrganisms)
        print("organism details: ")
        print("\t shape option: ", self.organismShape, "\t num features: ", self.numFeatures, "\t num genes: ", self.numGenes)
        print("evolution details: ")
        print("\t mutation rate: ", self.mutationRate, "\t scale factor: ", self.scaleFactor)
        print("\t spawn chance: ", self.spawnChance, "\t remove chance: ", self.removeChance)
        print("simulation details: ")
        print("\t steps: ", self.steps, "\t checkpoint: ", self.imgUpdate)
        print("output directory: ", self.outputDir)
        print("save file: ", self.save)
        print("checkpoint log: ", self.summary)


    def evolution(self):
        if os.path.exists(self.outputDir + self.save):
            if self.singleOrganism  == True:
                # single organism population does not have a load option, but can be recreated with past configs
                self.pop.currentBestOrganism = Organism(np.load(self.outputDir + self.save))
                self.pop.calcFitness(self.pop.currentBestOrganism)
            else:
                # the multiple organism population is more complex and has it's own load and save
                self.pop.loadOrganismData(self.outputDir + self.save)

            # get a sorted list of the images. start using the last image as the step for idxing
            filelist = [file for file in sorted(os.listdir(self.outputDir)) if file.endswith('.png')]
            num = filelist[-1].split('.')[0]
            self.start = int(num) * self.imgUpdate  # int converts from "XXX.png" to an int. mult by steps.
            self.i = self.start
            print("Loading from save file at: ", self.outputDir + self.save)
            print("starting from step: #", self.i)
        else:
            self.pop.spawn(numGenes=self.numGenes, numFeatures=self.numFeatures)
            self.start = 0

        self.loop()

    def loop(self):
        if self.start <= self.i < self.steps:
            self.pop.step(self.i, mutationRate=self.mutationRate, scaleFactor=self.scaleFactor,
                          spawnChance=self.spawnChance, removeChance=self.removeChance)
            if self.i % 100 == 0:
                print("step # ", self.i)
        else:
            #throw error message here
            if self.i >= self.steps:
                print("Program ran ", self.steps, " and has finished simulation")
                #print full summary of where the simulation is at
                # num check pts, num genes, %simularity
                return
            pass
        self.i = self.i + 1
        wx.CallLater(25, self.loop) #1000 = 1 second

    def startSimulation(self):
        self.createPopulation()
        print("starting simulation")
        self.evolution()
