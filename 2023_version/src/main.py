##--------------------------------------------------------------------\
#   GA_Drawing_Example
#   'main.py'
#   Main file for driving project. Image evolver instance and children
#   modified from: https://cosmiccoding.com.au/tutorials/genetic_part_one
#
#   Author: Lauren Linkous (LINKOUSLC@vcu.edu)
#   February 01, 2023
##--------------------------------------------------------------------\

# python lib imports
import wx.lib.mixins.inspection as wit

# project file imports
from GFrame import GFrame               # wxpython GUI
from evolve_image import EvolveImage    # class that controls

def main():

    # path to reference image and the output directory for checkpoints and reports
    img = "src/images/pearl.png"
    outputDir = "src/output-pearl/"
    # title for GUI window
    title = "GA Drawing:" + img

    # GUI
    # create the wxpython based GUI window for displaying the organisms
    # The GUI is driven by the image evolver class
    # - it will update as evolution happens rather than on a timer
    app = wit.InspectableApp()
    GF = GFrame(None, title=title)
    # fit the initial image to the screen
    GF.setImageAndFit(img)
    GF.clearScreen()
    # get the scaled reference image copy
    refImage = GF.getRefImage()
    GF.Show()

    # image evolver
    # This drives the GUI.
    # inputs:
    # the reference image (to initialize scalable variables)
    # The GUI object (to draw to GUI without multithreading this program)

    EI = EvolveImage(refImage)
    EI.addGUI(GF)
    # set the evolution params - mutation rate, scale factor, gene spawn & remove chances
    EI.setEvolutionParams(mutationRate=0.03, scaleFactor=0.25, spawnChance=0.20, removeChance=0.03)
    # set the population - single/multiple organism options
    EI.setPopulationParams(singleOrg=True, numOrganisms=10)
    # set organism params - number of genes controls complexity and variability, shape of organism controls the feature space
    # organism shape options: 0 = circle, 1 = square, 2 = rectangle,
    # to set the length of the square, change self.square_length in population2.py at the top
    EI.setOrganismParams(numGenes=10, organismShape=1) #numFeatures=7 removed to configure for shape instead
    EI.setSimulationParams(steps=100000, imgUpdate=200, outputDir=outputDir, saveFile="save.json") #save.npy if single
    # print a summary of settings. This is useful if you're using the default options to view any soft error correction
    # that may have happened with mis-matched parameters. Priority of inputs is Evolution>Population>Organism>Simulation
    EI.printParameterSummary()
    # start the simulation - this will begin drawing in the gui window
    EI.startSimulation()

    # adding to main loop keeps the gui window open and the program running
    # must be at the end because code after this will not execute while gui window is open
    app.MainLoop()



if __name__ == '__main__':
    main()
