
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import glob
from scipy import signal


class Line:
    """
    The line class defines a bunch of characteristics of a single line (lane line)
    It also includes a function to return the curvature of the line.
    """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

        #Conversions from pixels to real measurements
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700

    def get_curvature(self, which_fit='best'):
        """
        Returns the curvature of the line.
        """
        
        if which_fit == 'best':
            fit = self.best_fit
        else:
            fit = self.current_fit

        y_eval = np.max(self.ally)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        fit_cr = np.polyfit(self.ally*self.ym_per_pix, 
                            self.allx*self.xm_per_pix, 2)
        
        #Radius of curvature formula.
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return self.radius_of_curvature


class HistogramLineFitter:
    """
    The HistogramLineFitter uses an adaptive histogram fitting technique to 
    determine where the lanes most likely are.

    A line is defined as a yellow or white single line in traffic. A lane is a

    combinations of two of the lines.
    """

    def __init__(self):

        return

    def get_line(self, img, line, direction="left"):

        # Window dimensions for histograms sliding window.
        # see _get_histogram method for ye,ys,xs, and xe explainations
        
        win_width = 25
        win_height = 50 

        if not line.detected:

            xm = img.shape[1]
            ym = img.shape[0]
            h = self.__get_histogram(img, int(ym*(.5)), ym, 0, xm)


            # Find both peaks
            peaks = signal.find_peaks_cwt(h, np.arange(100,200))
            if direction == 'left':
                peak = peaks[0]
            else:
                peak = peaks[-1]
            
            # Move the sliding window and gather the associated points.
            yvals = []
            xvals = []
            
            for i in range(win_height):

                if direction == 'left':
                    if peak < win_width:
                        peak = win_width
                else:
                    if peak >= (xm - win_width):
                        peak = xm - win_width - 1
                
                start_range = int(ym*((win_height-i-1) / win_height))
                end_range = int(ym*((win_height-i) / win_height))

                for yval in range(start_range , end_range):
                    for xval in range(peak-win_width, peak + win_width):
                        if img[yval][xval] == 1.0:
                            yvals.append(yval)
                            xvals.append(xval)
                # Find new peaks to move the window for next iteration
                # new peaks will be the max in the current window plus 
                # the beginning of the window...
                
                ## See __get_histogram function for explaination.
                ye = int(ym *((win_height-i-1)/win_height)) 
                ys = int(ym *((win_height-i)/win_height))
                xs = int(peak-win_width)
                xe = int(peak+win_width)

                h = self.__get_histogram(img, ye, ys, xs, xe)
                if len(signal.find_peaks_cwt(h, np.arange(100,200))) > 0:
                    peak = np.amax(signal.find_peaks_cwt(h, np.arange(100,200))) + xs
                
                else: 
                # Look in bigger window
                    win_width_big = 100
                    ye = int(ym*((win_height-i-1)/win_height))
                    ys = int(ym*((win_height-i)/win_height))
                    xs = int(peak-win_width_big)
                    xe = int(peak+win_width_big)

                    h = self.__get_histogram(img, ye, ys, xs, xe)

                    if len(h > 0):
                        if len(signal.find_peaks_cwt(h, np.arange(100,200))) > 0:
                            peak = np.amax(signal.find_peaks_cwt(h, np.arange(100,200))) + xs

            yvals = np.asarray(yvals)
            xvals = np.asarray(xvals)
           
            line.allx = xvals
            line.ally = yvals
            
            # Fit a second order polynomial to lane line
            fit = np.polyfit(yvals, xvals, 2)
            
            line.current_fit = fit
            line.best_fit = fit

            
            fitx = fit[0]*yvals**2 + fit[1]*yvals + fit[2]

            
            line.recent_xfitted.append(fitx)
            line.bestx = fitx
            
        else:
            #initial peak - use previous line x
            peak = line.bestx[0]
            prev_line = copy(line)
            
            #move the sliding window across and gather the points
            yvals = []
            xvals = []
            
            for i in range(win_height):
                #peaks may be at the edge so we need to stop at the edge
                if direction == 'left':
                    if int(peak) < win_width:
                        peak = win_width
                else:
                    if int(peak) >= (xm - win_width):
                        peak = xm - win_width - 1
                        
                start_range = int(ym*((win_height-i-1)/win_height))
                end_range = int(xm*((win_height-i)/win_height))

                for yval in range(start_range, end_range):
                    for xval in range(int(peak-win_width), int(peak+win_width)):
                        if img[yval][xval] == 1.0:
                            yvals.append(yval)
                            xvals.append(xval)
                #use bestx to keep going over the line
                peak = line.bestx[(i + 1)%len(line.bestx)]

            yvals = np.asarray(yvals)
            xvals = np.asarray(xvals)
            
            line.allx = xvals
            line.ally = yvals
            
            # Fit a second order polynomial to lane line
            fit = np.polyfit(yvals, xvals, 2)
            line.current_fit = fit
            fitx = fit[0]*yvals**2 + fit[1]*yvals + fit[2]
            
            is_ok = self.__check_detection(prev_line, line)
            if is_ok:
                if len(line.recent_xfitted) > 10:
                    #remove the first element
                    line.recent_xfitted.pop(0)
                    line.recent_xfitted.append(fitx)
                    line.bestx = fitx
                    line.best_fit = fit
            else:
                # Line lost, go back to sliding window
                line.detected = false
            
        return line


    def __get_histogram(self, img, y_end, y_start, x_start, x_end):
        """
        Returns a histogram in the given windows. The images have the y axis pointing down
        of the z-axis pointing into the screen. The y_end, and x_end are the larger pixel limits
        of the window of the histogram.
                        |
        y_start-->  120 |
                        |
        y_end-->   360  |
                        |___________________________________________
                                  ^               ^
                               x_start (200)      x_end (400)
        """

        return np.sum(img[y_end:y_start , x_start:x_end], axis=0)

    def __check_detection(self, prev_line, next_line):
        """
        Checks two lines to see if they have similar curvature.
        """

        left_c = prev_line.get_curvature(which_fit='current')
        right_c = next_line.get_curvature(which_fit='current')
        # Checking that they are separated by approximately the right distance horizontally
        left_x = prev_line.recent_xfitted[0][0]
        right_x = next_line.recent_xfitted[0][0]
        if (np.absolute(left_x - right_x) > 1000) | (np.absolute(left_c - right_c) > 100):
            prev_line.detected = False
            next_line.detected = False
            return False

        prev_line.detected = True #in case these are different lines that are being compared
        next_line.detected = True
        return True


class LaneDrawer:
    """
    The tool used to draw lanes on the original image.
    """

    def __init__(self):
        
        self.center_offsets = []

        return

    def draw_lanes(self, undist, warped, lines, Minv, include_stats=False):
        """
        Takes in an image, a warped image, a Minv, some lines and draws stats and
        a fillPoly between the lines.
        """
        undist = np.copy(undist)
        img = np.copy(warped)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts = self.__get_xy_points(lines)
        stats = self.__get_lane_stats(lines, undist)


        # Draw the fill on the color_warp image
        color_warp = self.__draw_colored_fill(color_warp, np.absolute(stats['center_offset']), pts)
        
        color_warp = self.__draw_lane_pixels(lines['left_line'], color_warp, color='red')
        color_warp = self.__draw_lane_pixels(lines['right_line'], color_warp, color = 'blue')

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        if include_stats:
            add_stats = self.__write_statistics(result, stats)
            return add_stats
        else:        
            return result

    def __draw_colored_fill(self, img, offset, pts):
        """
        Draws a cv2.fillPoly that is colored according to how far it is away from the 
        center of the lane. Good for drivers to see how safe autonomous driving is!
        """
        limits = [0.45, 0.70]
        scale_factor = 255/((limits[1] - limits[0])/2)
        mid = (limits[0] + limits[1])/2

        if offset < mid:
            r = scale_factor *(offset - limits[0])
            cv2.fillPoly(img, np.int_([pts]), (r, 255, 0))

        elif (offset > mid) & (offset < limits[1]):
            g = scale_factor *(limits[1] - offset) 
            cv2.fillPoly(img, np.int_([pts]), (255, g, 0))
        else:
            cv2.fillPoly(img, np.int_([pts]), (255,0, 0))

        return img
        

    def __get_offset_average(self, new_offset, n=5):
        """
        Finds a running average of the center offsets. 
        """
        if len(self.center_offsets) > n:
            self.center_offsets.append(new_offset)
            self.center_offsets.pop(o)
            return (sum(self.center_offsets[-n:]) / n)
        else:
            return new_offset

    def __get_xy_points(self, lines):
        """Recast the x and y points into usable format for cv2.fillPoly()."""

        pts_left = np.array([np.transpose(np.vstack([lines['left_line'].allx,
                                                     lines['left_line'].ally]))])

        pts_right = np.array([np.flipud(np.transpose(np.vstack([lines['right_line'].allx,
                                                                lines['right_line'].ally])))])
        
        return np.hstack((pts_left, pts_right))

    def __draw_lane_pixels(self, line, img, color='red'):
        """
        Draws the pixels associated with the allx and ally coordinates in the line.

        Change the colour with the tuplet.
        """
        if color == 'red':
            for idx,pt in enumerate(line.ally):
                cv2.circle(img,(line.allx[idx], pt), 2, (255,0,0), -1)
        if color == 'blue':
            for idx,pt in enumerate(line.ally):
                cv2.circle(img,(line.allx[idx], pt), 2, (0,0,255), -1)            

        return img
        
    def __get_center_offset(self, img, lines):
        """
        Returns the distance from the center of the lane, takes in lines dictionary
        and an image. Computes a running average of the last n values to smooth.
        """
        mid_poly = (lines['right_line'].bestx[0] - lines['left_line'].bestx[0]) / 2

        midpoint = img.shape[0] / 2
        
        diff_in_pix = midpoint - mid_poly
        #convert to meters
        xm_per_pix = 3.7/700
        result = diff_in_pix * xm_per_pix

        lines['left_line'].line_base_pos = result
        lines['right_line'].line_base_pos = result
        
        return self.__get_offset_average(result, n=10)
        
    def __get_lane_stats(self, lines, undist):
        """
        Returns the statistics for the lane. Takes in lines dictionary and an image.
        """
        left_curavature = lines['left_line'].get_curvature(which_fit='best')
        right_curvature = lines['right_line'].get_curvature(which_fit='best')
        average_curvature = int((left_curavature + right_curvature) / 2)
        center_offset = self.__get_center_offset(undist, lines)

        stats = {'average_curve': average_curvature, 
                 'center_offset': center_offset}

        return stats

    def __write_statistics(self, undist, stats):
        """
        Writes the statistics dictionary on the image.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        offset = stats['center_offset']

        if offset < 0:
            offset_text = 'Vehicle is ' + str(np.around(np.absolute(offset),2)) + 'm left of center'
        else:
            offset_text = 'Vehicle is ' + str(np.around(offset,2)) + 'm right of center'

        curve_text = 'Road radius of curvature: ' + str(np.around(stats['average_curve'],-1))  +' m' 
        
        cv2.putText(undist, curve_text,(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(undist, offset_text,(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)

        return undist

        

class ImageThresholder:
    """
    The ImageThresholder takes in an rgb image and spits out a thresholded image 
    using a variety of techniques. Filtering techniques aim to extract the yellow and 
    white traffic lines for a variety of conditions.
    """

    def __init__(self):
        return

    def __generate_color_spaces(self):
        self.hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        self.yuv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2YUV)
        self.gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)

    def get_thresholded_image(self, rgb):
        self.rgb = rgb
        self.__generate_color_spaces()
        gradx = self.__abs_sobel_thresh(orient='x', thresh=(10, 100))
        grady = self.__abs_sobel_thresh(orient='y', thresh=(5, 250))
        mag_binary = self.__mag_threshold(mag_thresh=(5, 100))
        dir_binary = self.__dir_threshold(dir_thresh=(0, np.pi/2))
        s_binary = self.__color_threshold_hsv("s", (120,255))
        v_binary = self.__color_threshold_yuv("v", (0,105))
        r_binary = self.__color_threshold_rgb("r", (230,255))
        self.thresh = np.zeros_like(dir_binary)

        #Combine results
        self.thresh[((gradx == 1) & (grady == 1)) | ((mag_binary == 1)
               & (dir_binary == 1)) & ((s_binary == 1))
               | ((v_binary ==1) | (r_binary == 1))] = 1

        return self.thresh


    def __color_threshold_hsv(self, channel="s", thresh=(170,255)):
        """Band pass filter for HSV colour space"""

        h, s, v = cv2.split(self.hsv)

        if channel == "h":
            target_channel = h
        elif channel == "l":
            target_channel = s
        else:
            target_channel = v

        binary_output = np.zeros_like(target_channel)
        binary_output[(target_channel >= thresh[0]) & (target_channel <= thresh[1])] = 1
        
        return binary_output


    def __color_threshold_rgb(self, channel="r", thresh=(170,255)):
        """Band pass filter for RGB colour space"""

        r,g,b = cv2.split(self.rgb)
        
        if channel == "r":
            target_channel = r
        elif channel == "g":
            target_channel = g
        else:
            target_channel = b

        binary_output = np.zeros_like(target_channel)
        binary_output[(target_channel >= thresh[0]) & (target_channel <= thresh[1])] = 1
        
        return binary_output

    def __color_threshold_yuv(self, channel="v", thresh=(0,255)):
        """Band pass filter for YUV colour space"""

        y, u, v  = cv2.split(self.yuv)
        
        if channel == "y":
            target_channel = y
        elif channel == "u":
            target_channel = u
        else:
            target_channel = v

        binary_output = np.zeros_like(target_channel)
        binary_output[(target_channel >= thresh[0]) & (target_channel <= thresh[1])] = 1
        
        return binary_output


    def __abs_sobel_thresh(self, orient='x', thresh=(0,255)):
        """Apply a Sobel filter to find edges, scale the results
        from 1-255 (0-100%), then use a band-pass filter to create a mask
        for values in the range [thresh_min, thresh_max].
        """
        sobel = cv2.Sobel(self.gray, cv2.CV_64F, (orient=='x'), (orient=='y'))
        abs_sobel = np.absolute(sobel)
        max_sobel = max(1,np.max(abs_sobel))
        scaled_sobel = np.uint8(255*abs_sobel/max_sobel)
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output


    def __mag_threshold(self, sobel_kernel=3, mag_thresh=(0, 255)):
        """
        Function that takes image, kernel size, and threshold and returns
        magnitude of the gradient
        """

        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        return binary_output


    def __dir_threshold(self, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
        """
        Function to threshold gradient direction in an image for a given 
        range and Sobel kernel.
        """

        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

        return binary_output


class DistortionCorrector:    
    """Takes in path to calibration files as string.

    Params:
    self.nx = How many inside corners in chess boards (y-direction)
    self.ny = How many inside corner in chess boards (x-direction)
    calibration_folder_path = path to calibration images.

    Methods:

    fit: Calibrates the distorsionCorrector with array of images [None, width, height, channels]
    undistort: takes in an image, and outputs undistorted image
    test: takes in an image, and displays undistored image alongside original.

    -----------
    In this project it is already fitted, however it can be used for other projects.

    To Fit:

    # cal_images_paths = glob.glob('./camera_cal/cal*.jpg')
    # cal_images = []
    # for fname in cal_images_paths:
    #     cal_images.append(mpimg.imread(fname))
    # distCorrector.fit(cal_images)

    """  
    def __init__(self, calibration_folder_path):

        # Set nx and ny according to how many inside corners in chess boards images.  
        self.nx = 9
        self.ny = 6
        self.mtx = []
        self.dist = []
        self.cal_folder = calibration_folder_path

        fname = self.cal_folder + 'calibration.p'

        if  os.path.isfile(fname):
            print('Loading saved calibration file...')
            self.mtx, self.dist = pickle.load( open( fname, "rb" ) )
        else:
            print('Mtx and dist matrix missing. Please call fit distortionCorrector')
        return

    def fit(self, images):
        """Calibrates using chess images from camera_cal folder. 
        Saves mtx and dist in calibration_folder_path
        """
        
        cname = self.cal_folder + 'calibration.p'
        if  os.path.isfile(cname):
            print('Deleting existing calibration files...')
            os.remove(cname)

        print("Computing camera calibration...")

        objp = np.zeros((self.ny*self.nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)

        objpoints = []
        imgpoints = [] 


        # Step through the list and search for chessboard corners
        for img in images:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        if not ret:
            raise ValueError('Most likely the self.nx and self.ny are not set correctly')

        img = images[0]

        # Calibrate the camera and get mtx, and dist matricies.
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints,
                                                 imgpoints,
                                                 img.shape[:-1],
                                                 None, None)

        pname = self.cal_folder + 'calibration.p'
        print("Pickling calibration files..")
        pickle.dump( (self.mtx, self.dist), open( pname, "wb" ) )

        return


    def undistort(self, img):
        """Returns undistored image"""

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


    def test(self, img):

        undist = self.undistort(img)
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image')

        plt.show()
        return




