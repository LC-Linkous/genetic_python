##--------------------------------------------------------------------\
#   GA_Drawing_Example
#   'GFrame.py'
#   Class for GUI layout and basic functionality
#   Author: Lauren Linkous (LINKOUSLC@vcu.edu)
#   November 19, 2022
##--------------------------------------------------------------------\

import copy
import wx #pip install wxpython
import wx.aui
import wx.lib.newevent
import wx.lib.mixins.inspection as wit
import numpy as np

#default frame/panel sizes
import constants as c

class GFrame(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent=parent, title=title, size=(c.WIDTH, c.HEIGHT))
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour("#E6E6E6")
        self.Bind(wx.EVT_PAINT, self.onPaint)

        self.mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.mainSizer.Add(self.panel,-1, wx.EXPAND)
        self.SetSizer(self.mainSizer)

        #image memory
        self.refImage = None
        self.img = None
        self.imageBitmap = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(c.PANEL_HEIGHT, c.PANEL_WIDTH))

        self.shapes = []
        self.shapeType = 0

    def setImageAndFit(self, img="images/starrynight.jpg"):
        # load in + scale image to use as reference
        tmpImg = wx.Image(img, wx.BITMAP_TYPE_ANY)
        self.img = tmpImg.Scale(c.PANEL_WIDTH, c.PANEL_HEIGHT, wx.IMAGE_QUALITY_HIGH)
        #convert the image to an numpy array
        arr = np.frombuffer(self.img.GetDataBuffer(), dtype='uint8', count=-1, offset=0)
        #reshape the array
        arr = arr.reshape(c.PANEL_HEIGHT, c.PANEL_WIDTH, 3)
        #make this the reference image
        self.refImage = copy.deepcopy(arr)
        #draw the image on the panel once
        self.imageBitmap.SetBitmap(self.img)
        #refresh the panel
        self.panel.Refresh()


    def clearScreen(self):
        # new blank bitmap to replace starter image
        # usually called once, after 'setImageAndFit'.
        self.img = wx.Bitmap(c.PANEL_HEIGHT, c.PANEL_WIDTH).ConvertToImage()

    def onClose(self, event):
        self.Destroy()

    def onPaint(self, event=None):
        # custom 'onPaint' event
        # takes optional event argument so that it can be called both by system and program
        #print("paint event called")
        if len(self.shapes) < 1:
            # this can be commented out if it prints out too often
            # it's a good indicator that the event is being called too often if there's
            # an issue with a new shape/algorithm update
            #print("paint event called, but nothing to draw")
            #Note: This event triggers far more often on linux systems than windows
            return

        bit = wx.Bitmap(c.PANEL_HEIGHT, c.PANEL_WIDTH)
        dc = wx.MemoryDC(bit) # to draw on the new bitmap
        dc.SetBackground(wx.Brush("White"))
        dc.Clear()
        for ci in self.shapes:
            # color is always the last value in the arr/tuple
            col = ci[-1]
            # set color
            brush = wx.Brush(col)
            dc.SetBrush(brush)
            # set draw type
            pen = wx.Pen(col)
            dc.SetPen(pen)
            # call wx library for predefined shape
            if self.shapeType == 0: # circle
                dc.DrawCircle(ci[0], ci[1], ci[2])
            elif self.shapeType == 1 or self.shapeType == 2:  # square or rectangle
                dc.DrawRectangle(ci[0], ci[1], ci[2], ci[3])
            else:
                print("ERROR: shape type not recognized in GFrame. not drawing")

        #clear shapes because everything has been drawn
        self.shapes = []
        dc.SelectObject(wx.NullBitmap)  # deselect out of memory
        self.imageBitmap.SetBitmap(bit)
        self.img = bit.ConvertToImage()

    def setShapeType(self, i=0):
        #an int representing the types of shapes that can be drawn
        # 0 = circle
        # 1 = square
        # 2  rectangle
        self.shapeType = i
        print("shape type set in GUI")

    def setShapeList(self, arr):
        #gets list of shape position, coords, + color to pass to draw
        self.shapes = arr

    def addShape(self, newShape):
        self.shapes.append(newShape)

    def getListOfShapes(self):
        return self.shapes

    def getBitmap(self):
        return self.imageBitmap

    def getCurrentImage(self):
        return self.img

    def getRefImage(self):
        return self.refImage

    def saveImage(self, filename="outputImage.bmp"):
        #saves the bitmap
        imgBit = wx.Bitmap(self.img)
        imgBit.SaveFile(filename, wx.BITMAP_TYPE_BMP)



if __name__ == "__main__":

    #test out the window and draw functionality
    app = wit.InspectableApp()
    GF = GFrame(None, title='GFRAME GA wx test')
    GF.setImageAndFit()
    GF.Show()
    arr = [[200,150,50, (0,0,32)],[20,500,105,(100,25,7)],
           [800,15,200,(0,255,32)],[450,300,75,(40,60,32)]]
    GF.setShapeList(arr)
    app.MainLoop()
