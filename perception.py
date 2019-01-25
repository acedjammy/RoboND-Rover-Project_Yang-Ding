import numpy as np
import cv2

rock_x_temp = [0]
rock_y_temp = [0]


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(170, 170, 170)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def find_rock(img, rgb_thresh=(110, 110, 50)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))# keep same size as input image
    return warped,mask


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    image = Rover.img
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])

    warped, mask = perspect_transform(Rover.img, source, destination)
    pas = color_thresh(warped)
    obs = abs(np.float32(pas)-1)*mask
    roc = find_rock(warped)

    pas_x,pas_y = rover_coords(pas)
    obs_x,obs_y = rover_coords(obs)   
    roc_x,roc_y = rover_coords(roc)  
    
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    
    pas_x_w,pas_y_w = pix_to_world(pas_x,pas_y, xpos, ypos, yaw, world_size, scale)
    obs_x_w,obs_y_w = pix_to_world(obs_x,obs_y, xpos, ypos, yaw, world_size, scale)
    roc_x_w,roc_y_w = pix_to_world(roc_x,roc_y, xpos, ypos, yaw, world_size, scale)

    if (roc_x_w.any()) and (roc_y_w.any()):
    	Rover.sample_nearby = 1
    	roc_dist, roc_angles = to_polar_coords(roc_x,roc_y)
    	Rover.samples_nav_angles = roc_angles
    	Rover.samples_nav_dist = roc_dist
    	mean_roc_dist = np.mean(roc_dist)
    	print(mean_roc_dist)
    	if mean_roc_dist < 5:
    		Rover.near_sample = 1
    elif not (roc_x_w.any()) and not (roc_y_w.any()):
    	Rover.sample_nearby = 0

    if not Rover.picking_up:
    	voting_matrix = np.zeros_like(Rover.worldmap[:,:,0])
    	voting_matrix = np.dstack((voting_matrix, voting_matrix, voting_matrix)).astype(np.float)
    	voting_matrix[pas_y_w,pas_x_w,2] += 500
    	voting_matrix[obs_y_w,obs_x_w,0] += 0.5
    	Rover.worldmap[roc_y_w,roc_x_w,:] =255
    	for y in range(0,199):
    		for x in range(0,199):
    			if voting_matrix[y,x,2] > voting_matrix[y,x,0] * 0.5:
    				Rover.worldmap[y,x,0] = 0
    				Rover.worldmap[y,x,2] = 255
    			elif voting_matrix[y,x,2] < voting_matrix[y,x,0]:
    				Rover.worldmap[y,x,0] +=20
    				Rover.worldmap[y,x,2] -=5
    				if Rover.worldmap[y,x,0] >= 255:
    					Rover.worldmap[y,x,0] = 255
    				if Rover.worldmap[y,x,2] <= 0:
    					Rover.worldmap[y,x,2] = 0    
    
    dist, angles = to_polar_coords(pas_x,pas_y)
    Ndist, Nangles = to_polar_coords(obs_x,obs_y)

    mean_dir = np.mean(angles)-np.mean(Nangles)
    mean_dist = np.mean(dist)
    Rover.nav_angles = angles
    Rover.nav_dists = dist
    
    return Rover
