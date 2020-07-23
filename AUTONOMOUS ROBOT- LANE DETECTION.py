import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import RPi.GPIO as gpio
import time

def init():
    gpio.setmode(gpio.BOARD)
    gpio.setup(31, gpio.OUT) #IN1
    gpio.setup(33, gpio.OUT) #IN2
    gpio.setup(35, gpio.OUT) #IN3
    gpio.setup(37, gpio.OUT) #IN4
    gpio.setup(7 , gpio.IN, pull_up_down = gpio.PUD_UP)
    gpio.setup(12 , gpio.IN, pull_up_down = gpio.PUD_UP)
    
def gameover():
    init()
    #set all pins low
    gpio.output(31, False)
    gpio.output(33, False)
    gpio.output(35, False)
    gpio.output(37, False)

def forward():
    init()
    counterBR = np.uint64(0)
    buttonBR = int(0)
    counterFL = np.uint64(0)
    buttonFL = int(0)
    pwm2 = gpio.PWM(31,50)
    pwm = gpio.PWM(37,50)
    val = 22
    pwm2.start(val)
    pwm.start(val)
    time.sleep(0.1)
    e1_prev = 0
    e2_prev = 0
    kp = 0.1
    kd = 0.1
    ki = 0.02
    ts = 0.9
    e_prev_error = 0
    e2_prev_error = 0
    e_sum_error = 0
    e2_sum_error = 0
        
    print("counterFL = ",counterFL, "Gpio FL = ",gpio.input(7),"counter BR= ",counterBR, "Gpio BR= ",gpio.input(12))
    if int(gpio.input(7)) != int(buttonFL):
        buttonFL = int(gpio.input(7))
        counterFL += 1
            
    if int(gpio.input(12)) != int(buttonBR):
        buttonBR = int(gpio.input(12))
        counterBR += 1
            
    e_error = counterBR - counterFL
    time.sleep(0.1)
    e_prev_error = e_error   
    val += (e_error * kp) + (e_prev_error * kd) + (e_sum_error * ki)
    val = max(min(30,val),0)
    pwm2.ChangeDutyCycle(val)
    pwm.ChangeDutyCycle(val)
    e_sum_error += e_error 
 


def left():
    init()
    counterBR = np.uint64(0)
    buttonBR = int(0)
    counterFL = np.uint64(0)
    buttonFL = int(0)
    pwm2 = gpio.PWM(33,50)
    pwm = gpio.PWM(37,50)
    val = 40
    pwm2.start(val)
    pwm.start(val)
    time.sleep(0.1)
    e1_prev = 0
    e2_prev = 0
    kp = 0.1
    kd = 0.1
    ki = 0.02
    ts = 0.9
    e_prev_error = 0
    e2_prev_error = 0
    e_sum_error = 0
    e2_sum_error = 0

    print("counterFL = ",counterFL, "Gpio FL = ",gpio.input(7),"counter BR= ",counterBR, "Gpio BR= ",gpio.input(12))
    if int(gpio.input(7)) != int(buttonFL):
        buttonFL = int(gpio.input(7))
        counterFL += 1
            
    if int(gpio.input(12)) != int(buttonBR):
        buttonBR = int(gpio.input(12))
        counterBR += 1
            
    e_error = counterBR - counterFL
    time.sleep(0.1)
    e_prev_error = e_error   
    val += (e_error * kp) + (e_prev_error * kd) + (e_sum_error * ki)
    val = max(min(40,val),0)
    pwm2.ChangeDutyCycle(val)
    pwm.ChangeDutyCycle(val)
    e_sum_error += e_error 



def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 30
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    #plt.plot(histogram)
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    return out_img, ploty, left_fit, right_fit

def draw(image, thresh, left_fit, right_fit, Minv):
    warp_zero = np.zeros_like(thresh).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, thresh.shape[0]-1, thresh.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (thresh.shape[1], thresh.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result, color_warp, leftx, rightx

def generate_data(ym, xm, ploty, left_fit, right_fit):
    np.random.seed(0)
    ploty = np.linspace(0, thresh.shape[0]-1, thresh.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
    # Fit a second order polynomial to pixel positions in each fake lane line
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym, leftx*xm, 2)
    right_fit_cr = np.polyfit(ploty*ym, rightx*xm, 2)
    
    return ploty, left_fit_cr, right_fit_cr

def measure_curvature_real(ploty, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym = 30/720 # meters per pixel in y dimension
    xm = 3.7/700 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty, left_fit_cr, right_fit_cr = generate_data(ym, xm, ploty, left_fit, right_fit)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def turn(center, leftx, rightx):
    mean = leftx + (rightx - leftx)/2
    offset = (center - (mean*-1))/100
    offset = float(offset[0])
    return offset

image1 = cv2.imread('test.png')
cap = cv2.VideoCapture(0)
_, frame = cap.read()
rows, cols, _ = frame.shape
x_medium = int(cols / 2)
center1 = int(cols / 2.78)
while True:
    _, image1 = cap.read()
    
    k = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])

    image2 = cv2.undistort(image1, k, dist, None, k)
    image = cv2.resize(image2, (480,240))
    image = cv2.flip(image, -1)
    
    height = image.shape[0]
    width = image.shape[1]
    # Vertices coordinates in the source image
    s1 = [width // 20 - 100, height*0.5 ]
    s2 = [width // 1 + 100, height*0.5 ]
    s3 = [-100, height*0.4]
    s4 = [width + 100 , height*0.4]
    src = np.float32([s1, s2, s3, s4])
    # Vertices coordinates in the destination image
    d1 = [100, 0]
    d2 = [width - 100, 0]
    d3 = [100, height]
    d4 = [width - 100, height]
    distance = np.float32([d1, d2, d3, d4])
    M = cv2.getPerspectiveTransform(src, distance)
    # Warp the image
    pers = cv2.warpPerspective(image, M, (width, height))
    # We also calculate the oposite transform
    Minv = cv2.getPerspectiveTransform(distance, src)
    hsv = cv2.cvtColor(pers, cv2.COLOR_BGR2HSV)
    low2 = np.array([86,0,200])
    high2 = np.array([200, 255, 255])
    thresh = cv2.inRange(hsv, low2, high2)

    center = int(thresh.shape[0]) / 2
    out_img, ploty, left_fit, right_fit = fit_polynomial(thresh)
    result, color_warp, leftx, rightx = draw(image, thresh, left_fit, right_fit, Minv)
    left_curverad, right_curverad = measure_curvature_real(ploty, left_fit, right_fit)
    radius = np.mean([left_curverad, right_curverad])
    #cv2.putText(result, 'Radius of Curvature: {} m'.format(round(radius)), (0, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    offset = turn(center, leftx, rightx)
    offset = round(offset)
    center = cv2.line(result, (center1, 200), (center1, 270), (0,255,0), 2)
    cv2.putText(result, 'offset: {} m'.format(round(offset)), (150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow("result",result)
    cv2.waitKey(1) 

    if(offset > 3 and offset < 5):
        
        #cv2.putText(result, 'turn: {} m'.format(round(offset)), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 4)
        cv2.putText(result, 'straight', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        forward()

    elif(offset < 4):

     #   cv2.putText(result, 'turn: {} m'.format(round(offset)), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 4)
        cv2.putText(result, 'left', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        left()

    #plt.imshow(result)
    #plt.show()
gpio.cleanup()
    #cv2.imshow("result",result)
