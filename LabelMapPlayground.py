"""Labels the map"""

from PIL import Image
from PIL import ImageFont, ImageDraw, ImageOps
from Map import Map
from MapUtils import Coordinate

my_map = Map.pheonix()
im = my_map.image
print my_map.top_right_coord.longitude
my_map.add_label_to_image("middle", None, Coordinate((my_map.top_left_coord.latitude+my_map.bottom_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/2.0), False, 1.0)

im.show()

