# from openslide import open_slide
# import openslide
# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
# import cv2
# from openslide.deepzoom import DeepZoomGenerator

# # Load slide file
# slide = open_slide("FirstSlide.svs")

# # Generate objects for tiles using the Deep Zoom Generator
# tiles = DeepZoomGenerator(slide, tile_size = 256, overlap = 0, limit_bounds = False)
# # Here, we have divided our svs into tiles of size 256 with no overlap. 

# #The tiles object also contains data at many levels. 
# #To check the number of levels
# print("The number of levels in the tiles object are: ", tiles.level_count)

# print("The dimensions of data in each level are: ", tiles.level_dimensions)

# #Total number of tiles in the tiles object
# print("Total number of tiles = : ", tiles.tile_count)

# #How many tiles at a specific level?
# level_num = 15
# print("Tiles shape at level ", level_num, " is: ", tiles.level_tiles[level_num])
# print("This means there are ", tiles.level_tiles[level_num][0]*tiles.level_tiles[level_num][1], " total tiles in this level")

# #Dimensions of the tile (tile size) for a specific tile from a specific layer
# tile_dims = tiles.get_tile_dimensions(11, (3,4)) #Provide deep zoom level and address (column, row)


# #Tile count at the highest resolution level (level 15 in our tiles)
# tile_count_in_large_image = tiles.level_tiles[15] 
# #Check tile size for some random tile
# tile_dims = tiles.get_tile_dimensions(15, (20,20))
# #Last tiles may not have full 256x256 dimensions as our large image is not exactly divisible by 256
# tile_dims = tiles.get_tile_dimensions(15, (20,20))


# single_tile = tiles.get_tile(15, (62, 70)) #Provide deep zoom level and address (column, row)
# single_tile_RGB = single_tile.convert('RGB')
# single_tile_RGB.show()

# ###### Saving each tile to local directory
# cols, rows = tiles.level_tiles[15]

# import os
# tile_dir = "images/"
# for row in range(rows):
#     for col in range(cols):
#         tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
#         print("Now saving tile with title: ", tile_name)
#         temp_tile = tiles.get_tile(15, (col, row))
#         temp_tile_RGB = temp_tile.convert('RGB')
#         temp_tile_np = np.array(temp_tile_RGB)
#         plt.imsave(tile_name + ".png", temp_tile_np)
