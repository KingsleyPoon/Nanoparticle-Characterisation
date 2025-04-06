'''
    Version 2.
        Major changes made:
        1) Updated interface (application view)
        2) NP images can now be selected by through a Browse feature
        3) Shape parameter measurements can be adjusted by the image scale
        4) Removed generated figures
        5) Allow for multiple images to be selected
    
        Minor changes made:
        1)Feret package has been directly imported into the code
'''

import scipy
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.measure import label, regionprops
import cv2
import tkinter as tk
from tkinter import filedialog, font, ttk
from PIL import Image, ImageTk
import threading
from numpy.linalg import norm
import copy
import time


class NPCharacterizationApp:
    def __init__(self, root):
        
        self.root = root
        self.root.title("NP Characterization Application (Serial)")
        self.root.geometry("600x450")
        root.resizable(False, False)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-topmost", False)
        
        # Instance variable
        self.img_label = None
        self.mask = None
        self.regionOfInterest = None
        self.dimensionsarray = np.array([["MinFeret", "Orthogonal Length", "Radius"]])
        self.file_selected_text = "None"
        self.stop_analysis_event = threading.Event()
        
        
        # GUI components 
        self.open_button = tk.Button(self.root, text="Open File(s)", command=self.open_file_dialog, font = ("Helvetica",9))
        self.open_button.place(relx = 0.85, rely = 0.12, anchor='center', relwidth=0.23, relheight=0.10)
        
        self.analyze_button = tk.Button(root, text="Analyze Image", command=self.analyze_image, state=tk.DISABLED,font = font.Font(size = 9))
        self.analyze_button.place(relx = 0.85, rely = 0.25, anchor='center', relwidth=0.23, relheight=0.10)
        
        self.view_button = tk.Button(root, text="View Data", command=self.view_results,font = ("Helvetica",9))
        self.view_button.place(relx = 0.85, rely = 0.78, anchor='center', relwidth=0.23, relheight=0.10)
        
        self.path_label = tk.Label(self.root, text="File Selected: {}".format(self.file_selected_text))
        self.path_label.place(relx = 0.05,rely = 0.9)
        
        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.progress.place(anchor='center', relx=0.85, rely=0.67, relwidth=0.23, relheight=0.05)

        self.progress_label = tk.Label(self.root, text="0/0 Completed", font=("Helvetica", 10))
        self.progress_label.place(anchor='center',relx=0.85, rely=0.61)

        
        self.scale_label = tk.Label(self.root, text="Scale (px/nm)", font=("Helvetica", 9))
        self.scale_label.place(relx=0.73, rely=0.325)
        self.scale_number = tk.Entry(self.root, justify='right', width=5)  # Align text right
        self.scale_number.place(relx=0.88, rely=0.327, relwidth=0.08)
        self.scale_number.insert(0, "1")
        self.scale_number.config(validate="key")
        self.scale_number.config(validatecommand=(self.root.register(self.validate_input), '%P'))
        self.stored_scale_value = 1
       
        self.min_area = tk.Label(self.root, text="Min Area (nm²)", font=("Helvetica", 9))
        self.min_area.place(relx=0.73, rely=0.389)
        self.min_area_number = tk.Entry(self.root, justify='right', width=5)  # Align text right
        self.min_area_number.place(relx=0.89, rely=0.392, relwidth=0.07)
        self.min_area_number.insert(0, "1")
        self.min_area_number.config(validate="key")
        self.min_area_number.config(validatecommand=(self.root.register(self.validate_input), '%P'))

        
        self.max_area = tk.Label(self.root, text="Max Area (nm²)", font=("Helvetica", 9))
        self.max_area.place(relx=0.73, rely=0.452)
        self.max_area_number = tk.Entry(self.root, justify='right', width=10)  # Align text right
        self.max_area_number.place(relx=0.89, rely=0.455, relwidth=0.07)
        self.max_area_number.insert(0, "10000")
        self.max_area_number.config(validate="key")
        self.max_area_number.config(validatecommand=(self.root.register(self.validate_input), '%P'))

        self.circularity_min = tk.Label(self.root, text="Circ Min (0-1)", font=("Helvetica", 9))
        self.circularity_min.place(relx=0.73, rely=0.52)
        self.circularity_min = tk.Entry(self.root, justify='right', width=10)  # Align text right
        self.circularity_min.place(relx=0.89, rely=0.523, relwidth=0.07)
        self.circularity_min.insert(0, "0.01")
        self.circularity_min.config(validate="key")
        self.circularity_min.config(validatecommand=(self.root.register(self.validate_input), '%P'))
    
    #This function is required to ensure that a valid input is placed in the criteria cell (i.e., scale, NP area etc)
    def validate_input(self, new_value):
        # Allow only digits and one decimal point
        if new_value == "":
            return True
        try:
            # Try converting the input to a float
            float(new_value)
            return True
        except ValueError:
            return False

    #Function allows for viewing the results when you click the Results button        
    def view_results(self):
        # Create a new Toplevel window for the Treeview
        result_window = tk.Toplevel(self.root)
        result_window.title("View Data")
        result_window.geometry("600x400")  # Adjust window size for visibility

        # Extract headers (first row) and ensure they are strings
        headers = [str(header) for header in self.dimensionsarray[0]]
        tree = ttk.Treeview(result_window, show="headings")
        tree["columns"] = headers

        dimensionsarraycopy = copy.deepcopy(self.dimensionsarray)
        
        # Divide all values by the scale factor of image and format for one/two decimal places
        for row in dimensionsarraycopy[1:]:  # Skip the first row
            for i in range(3):  # Only the first three columns
                row[i] = round(float(row[i]) / float(self.scale_number.get()), 3)

        # Set up columns and headers
        for column in headers:
            tree.heading(column, text=column)
            tree.column(column, anchor='center', width=100)

        # Insert the remaining data (excluding the first row)
        for row in dimensionsarraycopy[1:]:
            row_values = [str(value) for value in row]
            tree.insert('', 'end', values=row_values)

        # Pack the Treeview
        tree.pack(expand=True, fill=tk.BOTH)

        # Add the Export Data button within the result window
        export_button = tk.Button(result_window, text="Export Data", command=lambda: self.copy_all_to_clipboard(tree))
        export_button.pack(pady=10)

        # Bind Ctrl+C to copy selected cells
        tree.bind("<Control-c>", self.copy_to_clipboard)
        # Bind Ctrl+A to select all rows
        tree.bind("<Control-a>", lambda event: self.select_all(tree))

    #Function that allows you to copy all results and paste into other programs such as excell
    def copy_to_clipboard(self, event):
        # Get all selected items
        selected_items = event.widget.selection()
        
        # Prepare a list to hold all copied data
        clipboard_data = []
    
        # Iterate through each selected item
        for item in selected_items:
            values = event.widget.item(item, 'values')
            clipboard_data.append("\t".join(values))
    
        # Join all lines with newline characters
        clipboard_data = "\n".join(clipboard_data)
    
        # Copy the combined data to the clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(clipboard_data)
        self.root.update()

    def copy_all_to_clipboard(self, tree):
        # Collect all data in the treeview
        all_data = []
        for child in tree.get_children():
            values = tree.item(child, 'values')
            all_data.append("\t".join(values))
        
        # Join all lines and copy to clipboard
        clipboard_data = "\n".join(all_data)
        self.root.clipboard_clear()
        self.root.clipboard_append(clipboard_data)
        self.root.update()
    
    def select_all(self, tree):
        # Clear any existing selections
        tree.selection_remove(tree.selection())
        # Select all rows
        for item in tree.get_children():
            tree.selection_add(item)
        
    # Function to open the file dialog and get the file path (Browse funcion)
    def open_file_dialog(self):
        file_paths = filedialog.askopenfilenames(title="Select a file", 
                                                    filetypes=[("Image Files", "*.tif; *.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if file_paths:
            try:
                self.file_paths = file_paths
                
                # Load and display the image
                img = Image.open(self.file_paths[0])
                self.update_image(img)
                
                #Update text to analyse                
                if len(file_paths) == 1:
                    self.path_label.config(text=f"File Selected: {self.file_paths[0].split('/')[-1]}")
                else:
                    self.path_label.config(text="File Selected: Multiple files selected.")

                #Enables analyze butto
                self.analyze_button.config(state=tk.NORMAL)
                
                self.file_selected_text = self.file_paths[0].split('/')[-1]
            except Exception as e:
                self.path_label.config(text="File Selected: Error loading image.")
                self.analyze_button.config(state=tk.DISABLED)


    def analyze_image(self):
        
        #Reset the flag that controls whether NPs are analysed or not
        self.stop_analysis_event.clear()
        
        #Ensure the values in the criteria boxes are valid
        if self.scale_number.get() == "" or float(self.scale_number.get()) == 0:
            self.scale_number.delete(0,tk.END)
            self.scale_number.insert(0,"1")
        
        if self.min_area_number.get() == "" or float(self.min_area_number.get()) == 0:
            self.min_area_number.delete(0,tk.END)
            self.min_area_number.insert(0,"1")
        
        if self.max_area_number.get() == "" or float(self.max_area_number.get()) == 0:
            self.max_area_number.delete(0,tk.END)
            self.max_area_number.insert(0,"1000000")
            
        if self.circularity_min.get() == "" or float(self.circularity_min.get()) > 1:
            self.circularity_min.delete(0,tk.END)
            self.circularity_min.insert(0,"1")
        
        self.min_area_stored = float(self.min_area_number.get())*float(self.scale_number.get())**2
        self.max_area_stored = float(self.max_area_number.get())*float(self.scale_number.get())**2
        
        
        if self.file_paths:
            # Start the analysis in a background thread (leaving main UI free to be used)
            threading.Thread(target=self.analyse_background_task, args=(self.file_paths,), daemon=True).start()
            self.cancel_button = tk.Button(root, text="Cancel Analysis", command=self.cancel_analysis,font = font.Font(size = 10))
            self.cancel_button.place(relx = 0.85, rely = 0.25, anchor='center', relwidth=0.23, relheight=0.10)
            self.open_button.config(state = tk.DISABLED)
    
    #Function to stop the analysis by setting the stop_analysis_event flag
    def cancel_analysis(self):
        self.stop_analysis_event.set()
        self.cancel_button.config(text = "Cancelling...",state=tk.DISABLED)

    #Function to determine the circularity of the NP    
    def calculate_circularity(self,region):
        if region.perimeter != 0:  # Avoid division by zero
            return (4 * np.pi * region.area) / (region.perimeter ** 2)
        else:
            return 0
    
    #Function to analyse the NP shape parameters
    def analyse_background_task(self,file_path):
        
        #Starting an array to store shape parameter values
        self.dimensionsarray = ["MinFeret","Orthogonal Length", "Radius","Circularity"]
        
        #Thresholding Parameters used to determine NP regions
        minimumArea = self.min_area_stored
        maximumArea = self.max_area_stored
        circularityThreshold = float(self.circularity_min.get())
        
        #Reseting progress bar values
        self.progress["maximum"] = 0
        self.progress["value"] = 0 
        
        #Variable to store regions to be processed
        regionList = []
        
        
        for file_path in self.file_paths:
            
            #Open image
            Im = plt.imread(file_path)
            img = Image.open(file_path)

    
            # If Image has more than one color channel, use only the first colour array
            # All 3 color channels should have agreement between zero and non-zero pixel values
            if len(np.shape(Im)) ==3:
                Im = Im[:,:,0]
            Im = Im>0
            Im = Im.astype('uint8')
    
    
            #Label regions of NPs in image
            labelimage = label(Im)
            unfiltered_regions = regionprops(labelimage)

            #Criteria to determine if region is considered NP or not and append to list
            regions = [region for region in unfiltered_regions 
                       if region.bbox[0] > 1
                       and region.bbox[1] > 1
                       and region.bbox[2] != Im.shape[0]
                       and region.bbox[3] != Im.shape[1]
                       and region.area > minimumArea
                       and region.area < maximumArea
                       and self.calculate_circularity(region) >= circularityThreshold]
            
            regionList.append(regions)
            
            #Update progress bar
            self.progress["maximum"] = len(regions) + self.progress["maximum"]

            
        self.root.after(0, self.update_gui_progressbar, self.progress["value"] ,self.progress["maximum"])
        
        #Loop to run through each NP region in list that is considered an appropriate NP
        for x, regions in enumerate(regionList):
            file_path =self.file_paths[x]
             
            '''
            These image read functions are being called up again as there may be more than one image analysed.
            The image to be analysed is determined by the variable x, as regions stored together in the previous loop
            '''
            Im = plt.imread(file_path)
            img = Image.open(file_path)

            if len(np.shape(Im)) ==3:
                Im = Im[:,:,0]
            Im = Im>0
            Im = Im.astype('uint8')
       
            #Label regions of NPs in image
            labelimage = label(Im)
            
            #Update the UI of which image is being analysed
            self.root.after(0,self.update_image(img))
            self.path_label.config(text=f"File Selected: {file_path.split('/')[-1]}")
            
            # For each region, ensure the stop_analysis_event flag (set by the cancel button) is not activated
            for i, region in enumerate(regions):
                if self.stop_analysis_event.is_set():
                    self.root.after(0,self.update_gui_buttons)
                    break

                # Identify size of smallest bounding box around region, transpose NP region intothe centre of a new image
                # Ensure there is sufficient blank area surrounding the NP  - approximately 20% larger than the maximum dimensions of the bounding box
                # Blank area ensures sufficient room for NP rotation in the image (during the differential evolution function) without exceeding the image boundaries
                yMinBox,xMinBox,yMaxBox,xMaxBox = region.bbox
                yLength = yMaxBox-yMinBox
                xLength = xMaxBox -xMinBox
         
                if yLength>xLength:
                    y = int(yLength*1.3)
                    self.regionOfInterest = np.zeros((y,y))
                else:
                    x = int(xLength*1.3)
                    self.regionOfInterest = np.zeros((x,x))
    
                size = np.shape(self.regionOfInterest)
                
                #This positions the region of interest at the centre of the 2D array
                self.regionOfInterest[int(size[0]/2-yLength/2):int(size[0]/2+yLength/2),int(size[0]/2-xLength/2):int(size[0]/2+xLength/2)] = Im[yMinBox:yMaxBox,xMinBox:xMaxBox]
                

                # Select only the largest element in bounding box for analysis (representing the NP)
                labelDuplicate = label(self.regionOfInterest)
                regionsDuplicate = regionprops(labelDuplicate)
    
                elementIndexToAnalyse = 0
                largestArea = 0
                for j in range(0,len(regionsDuplicate)):
                    if regionsDuplicate[j].filled_area > largestArea:
                        elementIndexToAnalyse = j
                        largestArea = regionsDuplicate[j].filled_area
                
                #Redraw the image to be analysed containing ONLY the NP we are interested in
                object = regionsDuplicate[elementIndexToAnalyse]
                self.regionOfInterest[:,:] = 0
                x = object.coords[:,0]
                y = object.coords[:,1]
                self.regionOfInterest[x,y]=1

                #To increase the efficiency of the fitting, guess the shape parameters based on on the current region properties
                AR = AR = object.major_axis_length / object.minor_axis_length
                self.feret_calculate(self.regionOfInterest)
                minFeretAngle = self.minf_angle*180/math.pi
                minFeret = self.minf
                centroidY,centroidX = object.centroid
                
    
                '''
                Boundaries set for the optimisation problem
                These boundaries are based on the initial parameters set and are not expected to deviate far away from these intial values
                The size of the boundaries can be set to alter the solution space
                Boundary conditions: 
                    1) CentroidY position, 
                    2) CentroidX position,
                    3) Minimum Feret diameter,
                    4)Orthogonal dimension,
                    5) Angle of rotation,
                    6) Truncation radius
                '''
                
                bnds = ((size[0]/2*0.9,size[0]/2*1.1),
                        (size[0]/2*0.9,size[0]/2*1.1),
                        (minFeret*0.95,minFeret*1.05),
                        (minFeret*0.95,minFeret*AR*1.05),
                        (0,180),
                        (0,minFeret/2*1.05))
                
                #Run the differential evolution function
                results = scipy.optimize.differential_evolution(self.object_min_function,strategy = 'best1bin',bounds = bnds,tol = 0.001,x0 = [centroidY,centroidX,minFeret,minFeret*AR,minFeretAngle,1],maxiter=200)
                
                #Create variables for the corresponding results
                radius = (int(results['x'][5]))
                dimension1 = int(results['x'][2])
                dimension2 = int(results['x'][3])
                
                #Determine from the side lengths which is the minimum feret diameter and the orthogonal length
                minFeretLength = min(dimension1,dimension2)
                orthogonalLength = max(dimension1,dimension2)
                    
                #Append to the existing array the shape characteristics of the individual NP
                self.dimensionsarray = np.vstack([self.dimensionsarray,[minFeretLength,orthogonalLength,radius,round(self.calculate_circularity(region),4)]])
               
                    
                #Update Progress bar
                self.progress["value"] = self.progress["value"] + 1
                self.root.after(0, self.update_gui_progressbar, self.progress["value"]  ,self.progress["maximum"])
                
        self.root.after(0,self.update_gui_buttons)
        
        
    def object_min_function(self, params):
        
        yMaskSize,xMaskSize = np.shape(self.regionOfInterest)
        
        #Define input variables
        centreY = int(params[0])
        centreX = int(params[1])
        width = int(params[2])
        length = int(params[3])
        r = int(params[5])
        theta = int(params[4])
        
        # Return a large cost function value if the radius guessed is larger than the either linear dimension
        # This prevents having radius of curvature that is larger than the either of the linear dimensions
        if (r*2 > length) or (r*2>width):
            return 1e30
        
        #Create mask size identical to input image
        self.mask = np.zeros((yMaskSize,xMaskSize))
            
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
        self.mask = np.any((circle1,circle2,circle3,circle4,rectangle1,rectangle2,rectangle3),axis=0)
        
        #Convert boolean to numeric (0 for no NP, 1 for NP)
        self.mask = self.mask.astype('uint8')
        
        #Rotate mask
        if theta !=0:
            rotationMatrix = cv2.getRotationMatrix2D((xMaskSize/2,yMaskSize/2),theta,1)
            self.mask = cv2.warpAffine(self.mask,rotationMatrix,(xMaskSize,yMaskSize),flags = cv2.INTER_CUBIC)
        
        Difference = np.sum(np.logical_xor(self.regionOfInterest,self.mask))
        return Difference**2
    
    #Updates the progress bar
    def update_gui_progressbar(self, progress_value,total):
        self.progress["value"] = progress_value
        self.progress_label.config(text="{}/{} Completed".format(progress_value, total))
    
    #Removes the cancel button and allows you to open new files again
    def update_gui_buttons(self):
        self.cancel_button.destroy()
        self.root.update_idletasks()
        self.open_button.config(state=tk.NORMAL)
    
    #Function to update the image in main UI    
    def update_image(self,img):
        # Get the current size of the GUI
        gui_width = self.root.winfo_width()
        gui_height = self.root.winfo_height()

        # Set maximum dimensions for the image (e.g., 50% of the GUI size)
        max_width = int(gui_width * 0.6)
        max_height = int(gui_height * 0.6)

        # Get original image dimensions
        original_width, original_height = img.size

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height

        # Calculate new dimensions based on aspect ratio and maximum size
        if original_width > max_width or original_height > max_height:
            if aspect_ratio > 1:  # Wider than tall
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:  # Taller than wide
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
        else:
            new_width = original_width
            new_height = original_height

        # Resize the image while maintaining the aspect ratio
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        if self.img_label is not None:
            self.img_label.destroy()

        self.img_label = tk.Label(self.root, image=img_tk)
        self.img_label.image = img_tk  # Keep reference to avoid garbage collection
        self.img_label.place(relx=0.38, rely=0.45, anchor='center')
        
    #Derived from the Feret package to calculate the approximate minimum feret distance and angle
    def feret_calculate(self, img):
        self.img = img.astype(float)
        self.edge = False
    
        self.find_convexhull()
        self.y0, self.x0 = ndimage.center_of_mass(self.hull)
        self.calculate_minferet()
        
    #Function part of Feret Package
    def find_convexhull(self):
    
    #Function part of Feret Package
        if self.edge:
            ys, xs = np.nonzero(self.img)
            new_xs = np.concatenate(
                (xs + 0.5, xs + 0.5, xs - 0.5, xs - 0.5, xs, xs + 0.5, xs - 0.5, xs, xs))
            new_ys = np.concatenate(
                (ys + 0.5, ys - 0.5, ys + 0.5, ys - 0.5, ys, ys, ys, ys + 0.5, ys - 0.5))
            new_ys, new_xs = (new_ys * 2).astype(int), (new_xs * 2).astype(int)
            self.hull = cv.convexHull(np.array([new_ys, new_xs]).T).T.reshape(2, -1).astype(float)
        else:
            self.hull = cv.convexHull(np.transpose(np.nonzero(self.img))).T.reshape(2, -1).astype(float)
    def calculate_minferet(self):
        """ Method calculates the exact minimum feret diameter.  The result is equal to imagejs minferet. """
        length = len(self.hull.T)

        Ds = np.empty(length)
        ps = np.empty((length, 3, 2))

        for i in range(length):
            p1 = self.hull.T[i]
            p2 = self.hull.T[(i + 1) % length]

            ds = np.abs(np.cross(p2 - p1, p1 - self.hull.T) / norm(p2 - p1))

            Ds[i] = np.max(ds)

            d_i = np.where(ds == Ds[i])[0][0]
            p3 = self.hull.T[d_i]
            ps[i] = np.array((p1, p2, p3))

        self.minf = np.min(Ds)

        minf_index = np.where(Ds == self.minf)[0][0]

        (y0, x0), (y1, x1), (y2, x2) = ps[minf_index]

        if x0 == x1:
            self.minf_angle = 0
        else:
            m = (y0 - y1) / (x0 - x1)
            t = y0 - m * x0
            self.minf_angle = np.arctan(m) + np.pi / 2

        self.minf_coords = np.array(((y0, x0), (y1, x1), (y2, x2)))

        if self.minf_angle < 0:
            self.minf_angle += np.pi

        if self.edge:
            self.minf /= 2.
            self.minf_coords /= 2.

        if x0 == x1:
            self.minf_t = self.minf_coords.T[0][2]
        else:
            self.minf_t = self.minf_coords.T[0][2] - np.tan(self.minf_angle) * self.minf_coords.T[1][2]



# Create the main window
root = tk.Tk()
app = NPCharacterizationApp(root)

# Start the GUI loop
root.mainloop()

