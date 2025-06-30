'''
    Version 4.
        Minor Changes:
            1) Updated objective function - drawing
            2) Fixed bug with cancelling processes

Note: Parallelisation of code was achieved using AI.
    
'''

import scipy
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.measure import label, regionprops
import tkinter as tk
from tkinter import filedialog, font, ttk
from PIL import Image, ImageTk
import threading
from numpy.linalg import norm
import copy
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import os
import time


class NPCharacterizationApp:
    def __init__(self, root):
        
        
        # Application GUI set-up
        self.root = root
        self.root.title("NP Characterization Application (Parallel Computation)")
        self.root.geometry("600x450")
        root.resizable(False, False)
        self.root.wm_attributes("-topmost", False)
        
        #Setup for parllelisation
        self.max_concurrent_tasks = os.cpu_count() - 1
        self.executor = ProcessPoolExecutor(max_workers = self.max_concurrent_tasks)
        
        # Instance variable
        self.img_label = None
        self.mask = None
        self.regionOfInterest = None
        self.dimensionsarray = np.array([["MinFeret", "Orthogonal Length", "Radius", "Circularity"]])
        self.file_selected_text = "None"
        self.stop_analysis_event = threading.Event()
        
        
        # GUI components 
        self.open_button = tk.Button(self.root, text="Open File(s)", command=self.open_file_dialog, font = ("Helvetica",9))
        self.open_button.place(relx = 0.85, rely = 0.12, anchor='center', relwidth=0.23, relheight=0.10)
        
        self.analyze_button = tk.Button(root, text="Analyze Image", command=self.start_analysis_thread, state=tk.DISABLED,font = font.Font(size = 9))
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

    #Validation of input    
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

    # Allows for results to be viewed in a table        
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
        
        # Divide all values by the scale factor of image and format for two decimal places
        for row in dimensionsarraycopy[1:]:  # Skip the first row
            for i in range(3):  # Only the first three columns
                row[i] = round(float(row[i]) / float(self.scale_number.get()), 3)
            
            row[3] = round(float(row[3]),3)
            #row[3] = round(float(row[3]) / float(self.scale_number.get()),4)

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
        
    #Allows results to be copied to clipboard
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
        
    # Function to open the file dialog and get the file path
    def open_file_dialog(self):
        file_paths = filedialog.askopenfilenames(title="Select a file", 
                                                    filetypes=[("Image Files", "*.tif; *.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if file_paths:
            try:
                self.file_paths = file_paths[0]
                # Load and display the image
                img = Image.open(self.file_paths)

                self.update_image(img)

                self.path_label.config(text=f"File Selected: {self.file_paths.split('/')[-1]}")
                self.analyze_button.config(state=tk.NORMAL)
                self.file_selected_text = self.file_paths.split('/')[-1]
            
            except Exception as e:
                self.path_label.config(text="File Selected: Error loading image.")
                self.analyze_button.config(state=tk.DISABLED)
                print(e)

    def start_analysis_thread(self):
        
        self.stop_analysis_event.clear()
        
        #Ensure the values in the threshold textboxes are valid inputs
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
        
        #Create the cancel button
        self.cancel_button = tk.Button(root, text="Cancel Analysis", command=self.cancel_analysis,font = font.Font(size = 10))
        self.cancel_button.place(relx = 0.85, rely = 0.25, anchor='center', relwidth=0.23, relheight=0.10)
        self.open_button.config(state = tk.DISABLED)        
        
        #Start the background Thread
        thread = threading.Thread(target = self.analyse_background_task)
        thread.start()

           
    # Applies thresholding to NP regions
    def identify_nanoparticles(self):
        
        #Thresholding Parameters used to determine NP regions
        minimumArea = self.min_area_stored
        maximumArea = self.max_area_stored
        circularityThreshold = float(self.circularity_min.get())
        self.progress["maximum"] = 0
        self.progress["value"] = 0 # Update to current region index

        
        Im = plt.imread(self.file_paths)

        # If Image has more than one color channel, use only the first colour array
        # All 3 color channels should have agreement between zero and non-zero pixel values
        if len(np.shape(Im)) ==3:
            Im = Im[:,:,0]
        Im = Im>0
        Im = Im.astype('uint8')
        
        self.regions = Im

        #Label regions of NPs in image
        labelimage = label(Im)
        unfiltered_regions = regionprops(labelimage)
 
        regions = [region for region in unfiltered_regions 
                   if region.bbox[0] > 1
                   and region.bbox[1] > 1
                   and region.bbox[2] != Im.shape[0]
                   and region.bbox[3] != Im.shape[1]
                   and region.area > minimumArea
                   and region.area < maximumArea
                   and self.calculate_circularity(region) >= circularityThreshold]

        return regions
    
    #Cancels analysis process
    def cancel_analysis(self):
        self.stop_analysis_event.set()
        self.cancel_button.config(text = "Cancelling...",state=tk.DISABLED)

    #Runs the analysis on the NPs
    def analyse_background_task(self):

        self.dimensionsarray = ["MinFeret","Orthogonal Length", "Radius","Circularity"]            
        
        nanoparticles = self.identify_nanoparticles()
        total_particles = len(nanoparticles)
        
        #Reset the progress bar
        self.progress["maximum"] = total_particles
        self.progress["value"] = 0
        self.root.after(0, self.update_gui_progressbar, self.progress["value"] ,self.progress["maximum"])
        self.futures = []  
        
        for nanoparticle in nanoparticles:
            
            if self.stop_analysis_event.is_set():
                break
            
            while len(self.futures) >= self.max_concurrent_tasks:
                
                # Wait for at least one task to complete before continuing
                completed, _ = concurrent.futures.wait(self.futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                # Remove completed futures from the list
                self.futures = [future for future in self.futures if future not in completed]
                
            nanoparticle_info = {"bbox": nanoparticle.bbox}
            
            future = self.executor.submit(analyse_single_nanoparticle, nanoparticle_info ,self.regions)
            self.futures.append(future)
            future.add_done_callback(self.handle_result)
                
        # Wait for all remaining tasks to finish and then update UI
        concurrent.futures.wait(self.futures)
        self.root.after(0, self.update_gui_buttons)
        self.end_time = time.time()

    def handle_result(self, future):
        try:
            result = future.result()  # This is safe in a done callback
            minFeretLength, orthogonalLength, radius, circularity = result
            
            # UI update
            self.root.after(0, self.process_analysis_result, minFeretLength, orthogonalLength, radius, circularity)
    
        except Exception as e:
            print("Error in worker process:", e)

    def process_analysis_result(self, minFeretLength, orthogonalLength, radius, circularity):
        # Update the dimensions array with the result
        self.dimensionsarray = np.vstack([
            self.dimensionsarray,
            [minFeretLength, orthogonalLength, radius, circularity]
        ])
    
        # Update the progress bar
        self.progress["value"] += 1
        self.update_gui_progressbar(self.progress["value"], self.progress["maximum"])


    def update_gui_progressbar(self, progress_value,total):
        self.progress["value"] = progress_value
        self.progress_label.config(text="{}/{} Completed".format(progress_value, total))
    
    def update_gui_buttons(self):
        self.cancel_button.destroy()
        self.root.update_idletasks()
        self.open_button.config(state=tk.NORMAL)
        
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
    
    def calculate_circularity(self,region):
        if region.perimeter != 0:  # Avoid division by zero
            return (4 * np.pi * region.area) / (region.perimeter ** 2)
        else:
            return 0        

#Function used to set up analysis for each NP region
def analyse_single_nanoparticle(nanoparticle_info,regions):
    
    #Unpack values from dict
    bbox = nanoparticle_info["bbox"]

    yMinBox, xMinBox, yMaxBox, xMaxBox = bbox
    yLength = yMaxBox - yMinBox
    xLength = xMaxBox - xMinBox
 
    if yLength > xLength:
        size = int(yLength * 1.3)
        regionOfInterest = np.zeros((size, size))
    else:
        size = int(xLength * 1.3)
        regionOfInterest = np.zeros((size, size))

    
    #This positions the region of interest at the centre of the 2D array
    regionOfInterest[
        int(size/2-yLength/2):int(size/2+yLength/2),
        int(size/2-xLength/2):int(size/2+xLength/2)
        ] = regions[yMinBox:yMaxBox,xMinBox:xMaxBox]
   
    # Select only the largest element in bounding box for analysis (representing the NP)
    labelDuplicate = label(regionOfInterest)
    regionsDuplicate = regionprops(labelDuplicate)

    # Find the largest element
    object = max(regionsDuplicate, key=lambda r: r.filled_area)

    # Redraw the regionOfInterest to contain only the desired nanoparticle
    regionOfInterest[:,:] = 0
    regionOfInterest[tuple(zip(*object.coords))] = 1

    AR = AR = object.major_axis_length / object.minor_axis_length
    
    feret_result = feret_calculate(regionOfInterest)
    minFeretAngle = feret_result["minf_angle"] * 180 / math.pi
    minFeret = feret_result["minf"]
    centroidY,centroidX = object.centroid
    

    # Boundaries set for the optimisation problem
    # These boundaries are based on the initial parameters set and are not expected to deviate far away from these intial values
    # The size of the boundaries can be set to alter the solution space
    # Boundary conditions : 1) CentroidY position, 2) CentroidX positio,n 3) Minimum Feret diameter, 4)Orthogonal dimension, 5) Angle of rotation, 6) Truncation radius
    bnds = ((size/2*0.9,size/2*1.1),
            (size/2*0.9,size/2*1.1),
            (minFeret*0.95,minFeret*1.05),
            (minFeret*0.95,minFeret*AR*1.05),
            (0,180),
            (0,minFeret/2*1.05))
    
    #Creating these variables outside differential evolution function to minimize redundant recalls of the same code.
    x_, y_ = np.meshgrid(np.arange(size), np.arange(size))
    mask = np.zeros((size,size))
        
    results = scipy.optimize.differential_evolution(
        lambda params: object_min_function(params,regionOfInterest.astype(np.uint8),x_,y_,mask),
        strategy = 'best1bin',
        bounds = bnds,
        tol = 0.001,
        x0 = [centroidY,centroidX,minFeret,minFeret*AR,minFeretAngle,1],
        maxiter=200)
            
    #Create variables for the corresponding results
    radius = (int(results.x[5]))
    dimension1 = int(results.x[2])
    dimension2 = int(results.x[3])
    
    minFeretLength = min(dimension1,dimension2)
    orthogonalLength = max(dimension1,dimension2)
    
    areaOfNP = minFeretLength * orthogonalLength - 4*radius**2 + (4*math.pi*radius**2)/3
    perimeterOfNP = minFeretLength*2 + orthogonalLength*2 - 8*radius + 4*math.pi*radius
    
    circularity = 4 * math.pi * areaOfNP/(perimeterOfNP**2)
    
    return minFeretLength,orthogonalLength,radius,circularity

#Cost Function
def object_min_function(params,regionOfInterest,x_,y_,mask):
    
    yMaskSize,xMaskSize = np.shape(regionOfInterest)
    centreY, centreX, width, length, theta, r = map(int, params)
    
    # Return a large cost function value if the radius guessed is larger than the either linear dimension
    # This prevents having radius of curvature that is larger than the either of the linear dimensions
    if (r*2 > length) or (r*2>width):
        return 1e30

    #Defining boundaries of truncated square/rectangle (starting from verticals left to right - xvalues, and bottom to top  - yvalues )(Figure 1c)
    x1Boundary = int(centreX - length/2)
    x2Boundary = int(centreX - length/2 + r)
    x3Boundary = int(centreX + length/2-r)
    x4Boundary = int(centreX + length/2)

    y1Boundary = int(centreY - width/2)
    y2Boundary = int(centreY + width/2)

    r_squared = r**2
    # Precomputed centers of the 4 corner circles
    xc1, yc1 = x3Boundary, y2Boundary - r  # circle1
    xc2, yc2 = x3Boundary, y1Boundary + r  # circle2
    xc3, yc3 = x2Boundary, y1Boundary + r  # circle3
    xc4, yc4 = x2Boundary, y2Boundary - r  # circle4

    #Draw the respective rectangles and circles and combine all the areas together
    circle1 = (x_ - xc1)**2 + (y_ - yc1)**2 <= r_squared
    circle2 = (x_ - xc2)**2 + (y_ - yc2)**2 <= r_squared
    circle3 = (x_ - xc3)**2 + (y_ - yc3)**2 <= r_squared
    circle4 = (x_ - xc4)**2 + (y_ - yc4)**2 <= r_squared
    rectangle1 = ((x_ >= x2Boundary) & (x_ <= x3Boundary) & (y_ >= y1Boundary) & (y_ <= y2Boundary))
    rectangle2 = ((x_ >= x1Boundary) & (x_ <= x4Boundary) & (y_ >= yc3) & (y_ <= yc1))
    
    mask = circle1 | circle2 | circle3 | circle4 | rectangle1 | rectangle2
    
    #Convert boolean to numeric (0 for no NP, 1 for NP)
    mask = mask.astype('uint8')
    
    #Rotate mask
    if theta !=0:
        rotationMatrix = cv.getRotationMatrix2D((xMaskSize/2,yMaskSize/2),theta,1)
        mask = cv.warpAffine(mask,rotationMatrix,(xMaskSize,yMaskSize),flags = cv.INTER_CUBIC)
    
    Difference = np.count_nonzero(np.bitwise_xor(regionOfInterest, mask))
    return Difference**2
        
'''
Derived from the Feret package to calculate the approximate minimum feret distance and angle.
These have been converted to a statless function to allow for parallelisation
'''
def find_convexhull(img, edge=False):
    if edge:
        ys, xs = np.nonzero(img)
        new_xs = np.concatenate(
            (xs + 0.5, xs + 0.5, xs - 0.5, xs - 0.5, xs, xs + 0.5, xs - 0.5, xs, xs))
        new_ys = np.concatenate(
            (ys + 0.5, ys - 0.5, ys + 0.5, ys - 0.5, ys, ys, ys, ys + 0.5, ys - 0.5))
        new_ys, new_xs = (new_ys * 2).astype(int), (new_xs * 2).astype(int)
        hull = cv.convexHull(np.array([new_ys, new_xs]).T).T.reshape(2, -1).astype(float)
    else:
        hull = cv.convexHull(np.transpose(np.nonzero(img))).T.reshape(2, -1).astype(float)
    return hull

def calculate_minferet(hull, edge=False):
    length = len(hull.T)
    Ds = np.empty(length)
    ps = np.empty((length, 3, 2))

    for i in range(length):
        p1 = hull.T[i]
        p2 = hull.T[(i + 1) % length]

        ds = np.abs(np.cross(p2 - p1, p1 - hull.T) / norm(p2 - p1))
        Ds[i] = np.max(ds)
        d_i = np.where(ds == Ds[i])[0][0]
        p3 = hull.T[d_i]
        ps[i] = np.array((p1, p2, p3))

    minf = np.min(Ds)
    minf_index = np.where(Ds == minf)[0][0]
    (y0, x0), (y1, x1), (y2, x2) = ps[minf_index]

    if x0 == x1:
        minf_angle = 0
    else:
        m = (y0 - y1) / (x0 - x1)
        minf_angle = np.arctan(m) + np.pi / 2

    if minf_angle < 0:
        minf_angle += np.pi

    if edge:
        minf /= 2.
        ps[minf_index] /= 2.

    minf_coords = ps[minf_index]
    minf_t = minf_coords.T[0][2] if x0 == x1 else minf_coords.T[0][2] - np.tan(minf_angle) * minf_coords.T[1][2]

    return {
        "minf": minf,
        "minf_angle": minf_angle,
        "minf_coords": minf_coords,
        "minf_t": minf_t
    }

def feret_calculate(img, edge=False):
    img = img.astype(float)
    hull = find_convexhull(img, edge=False)
    y0, x0 = ndimage.center_of_mass(hull)
    feret_data = calculate_minferet(hull, edge=False)
    feret_data["centre"] = (y0, x0)
    return feret_data


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # (Optional, but good for Windows + pyinstaller)
    
    root = tk.Tk()
    app = NPCharacterizationApp(root)
    root.mainloop()
