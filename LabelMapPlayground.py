"""Labels the map"""

from PIL import Image
from PIL import ImageFont, ImageDraw, ImageOps
from Map import Map
from MapUtils import Coordinate, Position
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


my_map = Map.pheonix()
# im = my_map.image
# print my_map.top_right_coord.longitude
# my_map.add_label_to_image("middle", None, Coordinate((my_map.top_left_coord.latitude+my_map.bottom_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/2.0), False, 1.0)
# my_map.add_label_to_image("bottom_right", None, my_map.bottom_right_coord, False, 1.0)
# my_map.add_label_to_image("top_left", None, my_map.top_left_coord, False, 1.0)
# my_map.add_label_to_image("top_right", None, my_map.top_right_coord, False, 1.0)
# my_map.add_label_to_image("bottom_left", None, my_map.bottom_left_coord, False, 1.0)
# my_map.add_label_to_image("1/2 left 1/2 down", None, Coordinate((my_map.top_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/2.0), False, 1.0)
# my_map.add_label_to_image("1/2 left 2/3 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/2.0), False, 1.0)
# my_map.add_label_to_image("1/3 left 1/3 down", None, Coordinate((my_map.top_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
# my_map.add_label_to_image("2/3 left 2/3 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
# my_map.add_label_to_image("1/3 left 2/3 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
# my_map.add_label_to_image("2/3 left 1/3 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
# my_map.add_label_to_image("2/3 left 1/2 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
# #my_map.add_label_to_image("1/3 left 1/2 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
# my_map.add_label_to_image("80 30", Position(80,20), Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
# my_map.add_label_to_image("80 30", Position(80,40), Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)

# im.show()

im = plt.imread(my_map.image_path)
implot = plt.imshow(im)
label = "hi"
plt.scatter(50,80)
prop = fm.FontProperties(fname="fonts/ProximaNova.TTF")
plt.annotate(label, xy=(50,80), xytext = (40, 70), fontsize=12, fontproperties=prop)
plt.annotate(label, xy=(50,80), xytext = (40, 90))

plt.show()