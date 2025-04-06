'''
    Version 1.
        This is a proof of concept for the iterative mask refinement method.
        It is recommended that this code is run in the IDE Spyder to ensure 
        thatthe graphs do not appear after each NP is analysed.
'''

import scipy
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.measure import label, regionprops
import feret
import cv2
from functools import partial

#Global variable for the segmented region to be accessed inside main code and function
global regionOfInterest

#Variables to collect useful information
dimensionsarray = ["MinFeret","Orthogonal Length", "Radius","Circularity","Shape"]

# Cost function to derive difference between the segmented mask and the predicted mask and the value is equal to the sum of squares
def object_min_function(x, xMaskSize,yMaskSize):
    
    #Define input variables
    centreY = int(x[0])
    centreX = int(x[1])
    width = int(x[2])
    length = int(x[3])
    r = int(x[5])
    theta = int(x[4])
    
    # Return a large cost function value if the radius guessed is larger than the either linear dimension
    # This prevents having radius of curvature that is larger than the either of the linear dimensions
    if (r*2 > length) or (r*2>width):
        return 1e30
    
    #Create global variables to create access of values outside the function
    global mask

    #Create mask size identical to input image
    mask = np.zeros((yMaskSize,xMaskSize))
        
    #Defining boundaries of truncated square/rectangle (starting from verticals left to right - xvalues, and bottom to top  - yvalues )(Figure 1c)
    x1Boundary = int(centreX - length/2)
    x2Boundary = int(centreX - length/2 + r)
    x3Boundary = int(centreX + length/2-r)
    x4Boundary = int(centreX + length/2)

    y1Boundary = int(centreY - width/2)
    y2Boundary = int(centreY + width/2)

    
    #Draw the respective rectangles and circles and combine all the areas together
    x_, y_ = np.meshgrid(range(yMaskSize), range(xMaskSize))
    circle1 =((x_-x3Boundary)**2 + (y_-y2Boundary+r)**2)<=r**2
    circle2 =((x_-x3Boundary)**2 + (y_-y1Boundary-r)**2)<=r**2
    circle3 =((x_-x2Boundary)**2 + (y_-y1Boundary-r)**2)<=r**2
    circle4 =((x_-x2Boundary)**2 + (y_-y2Boundary+r)**2)<=r**2
    rectangle1 = ((x_ >= x2Boundary) & (x_ <= x3Boundary) & (y_ >= y1Boundary) & (y_ <= y2Boundary))
    rectangle2 = ((x_ >= x3Boundary) & (x_ <= x4Boundary) & (y_ >= y1Boundary+r) & (y_ <= y2Boundary -r))
    rectangle3 = ((x_ >= x1Boundary) & (x_ <= x2Boundary) & (y_ >= y1Boundary+r) & (y_ <= y2Boundary -r))
    mask = np.any((circle1,circle2,circle3,circle4,rectangle1,rectangle2,rectangle3),axis=0)
    
    #Convert boolean to numeric (0 for no NP, 1 for NP)
    mask = mask.astype('uint8')
    
    #Mask rotation
    if theta !=0:
        rotationMatrix = cv2.getRotationMatrix2D((xMaskSize/2,yMaskSize/2),theta,1)
        mask = cv2.warpAffine(mask,rotationMatrix,(xMaskSize,yMaskSize),flags = cv2.INTER_CUBIC)
    
    #Find the difference between the mask and region of interest and return the difference^2
    Difference = np.sum(np.logical_xor(regionOfInterest,mask))
    return Difference**2

'''
# Add filepath here (i.e., C:\\User\\image.tif)
'''

fileToRead = 
Im = plt.imread(fileToRead)

# Reduce the dimensionality of the image (in situation where it is a binary RGB image)
if len(np.shape(Im)) ==3:
    Im = Im[:,:,0]
Im = Im>0
Im = Im.astype('uint8')


#Identify regions of NPs that are eight-connected
labelimage = label(Im)
regions = regionprops(labelimage)

# For each region (representing a possible NP), check if it meets the thresholding criteria for it to be fitted
for region in regions:
   
    ''' 
    Change these threshold parameter values to determine which NPs should be accepted
    (ie to remove background noise that might have encoded as NP)
    '''
    
    circularityThreshold = 0.5
    minimumArea = 1000
    maximumArea = 1000000
   
    #Determine the bounds of the identified NP region
    yMinBox,xMinBox,yMaxBox,xMaxBox = region.bbox
    
    #Calculate the circularity
    try:
        circularity = 4*math.pi*region.area/(region.perimeter*region.perimeter)
    except:
        circularity = 0
    
    #The thresholding criteria is applied here. If it does not meet thie criteria, it is not processed and is skipped
    if (yMinBox > 1 
        and xMinBox > 1
        and  xMaxBox!= Im.shape[1]
        and yMaxBox != Im.shape[0]
        and region.area > minimumArea
        and region.area < maximumArea
        and circularity >= circularityThreshold):

        # Identify size of smallest bounding box around region, transpose NP region intothe centre of a new image
        # Ensure there is sufficient blank area surrounding the NP  - approximately 20% larger than the maximum dimensions of the bounding box
        # Blank area ensures sufficient room for NP rotation in the image without exceeding the image boundaries
        yLength = yMaxBox-yMinBox
        xLength = xMaxBox -xMinBox
 
        if yLength>xLength:
            y = int(yLength*1.2)
            regionOfInterest = np.zeros((y,y))
        else:
            x = int(xLength*1.2)
            regionOfInterest = np.zeros((x,x))
        size = np.shape(regionOfInterest)
        
        #This positions the region of interest at the centre of the 2D array
        regionOfInterest[int(size[0]/2-yLength/2):int(size[0]/2+yLength/2),int(size[0]/2-xLength/2):int(size[0]/2+xLength/2)] = Im[yMinBox:yMaxBox,xMinBox:xMaxBox]
        
        
        #Add a plot of the original NP region into a subfigure
        f = plt.figure(figsize = (8,6),dpi=800)
        f.add_subplot(1,3,1)
        plt.imshow(regionOfInterest)

        # Select only the largest element in bounding box for analysis (representing the NP)
        labelDuplicate = label(regionOfInterest)
        regionsDuplicate = regionprops(labelDuplicate)

        elementIndexToAnalyse = 0
        largestArea = 0
        for i in range(0,len(regionsDuplicate)):
            if regionsDuplicate[i].filled_area > largestArea:
                elementIndexToAnalyse = i
                largestArea = regionsDuplicate[i].filled_area
        
        #Redraw the image to be analysed containing ONLY the NP we are interested in
        object = regionsDuplicate[elementIndexToAnalyse]
        regionOfInterest[:,:] = 0
        x = object.coords[:,0]
        y = object.coords[:,1]
        regionOfInterest[x,y]=1

        #Guess the initial parameters for optimisation using the region properties
        imageAttributes = feret.calc(regionOfInterest)
        AR = object.major_axis_length / object.minor_axis_length
        minFeretAngle = imageAttributes.minf_angle * 180/math.pi
        minFeret = imageAttributes.minf
        centroidY,centroidX = object.centroid
        
        #Used to set the prediction mask to be the same size as our input segmented mask
        yMaskSize,xMaskSize = np.shape(regionOfInterest)
        
        # Boundaries set for the optimisation problem
        # These boundaries are based on the initial parameters set and are not expected to deviate far away from these intial values
        # The size of the boundaries can be set to alter the solution space
        # Boundary conditions : 1) CentroidY position, 2) CentroidX position, 3) Minimum Feret diameter, 4) Orthogonal dimension, 5) Angle of rotation, 6) Truncation radius
        bnds = ((size[0]/2*0.9,size[0]/2*1.1),
                (size[0]/2*0.9,size[0]/2*1.1),
                (minFeret*0.95,minFeret*1.05),
                (minFeret*0.95,minFeret*AR*1.05),
                (0,180),
                (0,minFeret/2*1.05))
        
        #Run the differential evolution function to determine the best fitting nanocube model
        results = scipy.optimize.differential_evolution(object_min_function,args=[xMaskSize,yMaskSize],strategy = 'best1bin',bounds = bnds,tol = 0.001,x0 = [centroidY,centroidX,minFeret,minFeret*AR,minFeretAngle,1],maxiter=200)
        

        #Extract the shape parameters from the fitting
        radius = (int(results['x'][5]))
        dimension1 = int(results['x'][2])
        dimension2 = int(results['x'][3])
        
        #Determine the minFeret length and the orthogonal length (MinFeret * AR)
        if dimension1 <= dimension2:
            minFeretLength = dimension1
            orthogonalLength = dimension2
        else:
            minFeretLength = dimension2
            orthogonalLength = dimension2
        
        # Determine Shape based on the available 2D space of aspect ratio and truncation level
        # The specific values can be altered depending on the definition of the shape boundaries
        aspectRatio = round(orthogonalLength/minFeretLength,3)
        radiusRatio =round(radius/(minFeretLength/2),3)
        if radiusRatio <0.05: 
            if radiusRatio <= 0.02 and aspectRatio <= 1.02:
                shape = "Cube"
            elif aspectRatio < 1.05:
                shape = "Cube-like"
            else:
                if radiusRatio <= 0.02:
                    shape = "Cuboid"
                else:
                    shape = "Cuboid - Truncated Cuboid Transition"
        elif radiusRatio <0.96:
            if aspectRatio <= 1.02:
                shape = "Truncated Cube"
            elif aspectRatio >= 1.05:
                shape = "Truncated Cuboid"
            else:
                shape = "Truncated Cube - Truncated Cuboid Transition"
        else:
            if radiusRatio >= 0.98 and aspectRatio <=1.02:
                shape = "Sphere"
            elif aspectRatio < 1.05:
                shape = "Sphere-like"
            else:
                if radiusRatio >= 0.98:
                    shape = "Rod"
                else:
                    shape = "Rod - Truncated Cuboid Transition"

        #Append to the existing array the shape characteristics of the individual NP
        dimensionsarray = np.vstack([dimensionsarray,[minFeretLength,orthogonalLength,radius,circularity,shape]])

        #Create a 2D pixel array of the difference etween the segmented region and the predicted region
        Difference = regionOfInterest-mask
       
        #Create two more additional figures: 1) (segmented - prediction) mask 2) predicted mask
        #The left figure in the subplot is the segmented region
        #The right figure is the predicted mask
        #Centre figure is the subtracted there mask
        f.add_subplot(1,3,2)
        plt.imshow(Difference)
        f.add_subplot(1,3,3)
        plt.imshow(mask)
        plt.show()
