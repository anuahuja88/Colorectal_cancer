# from openslide import open_slide
# import openslide
# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
# import cv2

# # Load slide file
# # slide = open_slide("FirstSlide.svs")

# properties = slide.properties
# print(properties)

# print("Vendor is:", properties['openslide.vendor'])
# print("Pixel size of X in um is:", properties['openslide.mpp-x'])
# print("Pixel size of Y in um is:", properties['openslide.mpp-y'])

# #Objective used to capture the image
# objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
# print("The objective power is: ", objective)

# # get slide dimensions for the level 0 - max resolution level
# slide_dims = slide.dimensions
# print(slide_dims)

# #Get a thumbnail of the image and visualize in Pillow Image format
# slide_thumb_600 = slide.get_thumbnail(size=(600, 600))
# # slide_thumb_600.show()

# #Convert thumbnail to numpy array
# slide_thumb_600_np = np.array(slide_thumb_600)
# cv2.imshow("slide_thumb_600", slide_thumb_600_np)
# cv2.waitKey(0)

# #Get slide dims at each level. Remember that whole slide images store information
# #as pyramid at various levels
# dims = slide.level_dimensions

# num_levels = len(dims)
# print("Number of levels in this image are:", num_levels)

# print("Dimensions of various levels in this image are:", dims)

# #By how much are levels downsampled from the original image?
# factors = slide.level_downsamples
# print("Each level is downsampled by an amount of: ", factors)

# #Copy an image from a level
# level3_dim = dims[2]
# #Give pixel coordinates (top left pixel in the original large image)
# #Also give the level number (for level 3 we are providing a valueof 2)
# #Size of your output image
# #Remember that the output would be a RGBA image (Not, RGB)
# level3_img = slide.read_region((0,0), 2, level3_dim) #Pillow object, mode=RGBA

# #Convert the image to RGB
# level3_img_RGB = level3_img.convert('RGB')
# # level3_img_RGB.show()

# #Convert the image into numpy array for processing
# level3_img_np = np.array(level3_img_RGB)
# cv2.imshow("level3_img_RGB", level3_img_np)
# cv2.waitKey(0)

# #Return the best level for displaying the given downsample.
# SCALE_FACTOR = 32
# best_level = slide.get_best_level_for_downsample(SCALE_FACTOR)
# #Here it returns the best level to be 2 (third level)
# #If you change the scale factor to 2, it will suggest the best level to be 0 (our 1st level)

# print(f"Best level for downsample at scale factor: {SCALE_FACTOR} = {best_level}")