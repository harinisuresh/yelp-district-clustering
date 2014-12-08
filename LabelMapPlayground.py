"""Labels the map"""

from PIL import Image
from PIL import ImageFont, ImageDraw, ImageOps
from Map import Map
from MapUtils import Coordinate, Position

my_map = Map.pheonix()
im = my_map.image
print my_map.top_right_coord.longitude
my_map.add_label_to_image("middle", None, Coordinate((my_map.top_left_coord.latitude+my_map.bottom_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/2.0), False, 1.0)
my_map.add_label_to_image("bottom_right", None, my_map.bottom_right_coord, False, 1.0)
my_map.add_label_to_image("top_left", None, my_map.top_left_coord, False, 1.0)
my_map.add_label_to_image("top_right", None, my_map.top_right_coord, False, 1.0)
my_map.add_label_to_image("bottom_left", None, my_map.bottom_left_coord, False, 1.0)
my_map.add_label_to_image("1/2 left 1/2 down", None, Coordinate((my_map.top_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/2.0), False, 1.0)
my_map.add_label_to_image("1/2 left 2/3 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/2.0), False, 1.0)
my_map.add_label_to_image("1/3 left 1/3 down", None, Coordinate((my_map.top_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
my_map.add_label_to_image("2/3 left 2/3 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
my_map.add_label_to_image("1/3 left 2/3 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.bottom_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
my_map.add_label_to_image("2/3 left 1/3 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude + my_map.top_left_coord.latitude)/3.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
my_map.add_label_to_image("2/3 left 1/2 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_right_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
#my_map.add_label_to_image("1/3 left 1/2 down", None, Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
my_map.add_label_to_image("80 30", Position(80,20), Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)
my_map.add_label_to_image("80 30", Position(80,40), Coordinate((my_map.bottom_left_coord.latitude+my_map.top_left_coord.latitude)/2.0, (my_map.top_left_coord.longitude + my_map.top_left_coord.longitude + my_map.top_right_coord.longitude)/3.0), False, 1.0)


im.show()
