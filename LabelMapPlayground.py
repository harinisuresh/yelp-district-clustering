"""Labels the map"""

from PIL import Image
from PIL import ImageFont, ImageDraw, ImageOps
from Map import Map
from MapUtils import Coordinate

my_map = Map.pheonix()
im = my_map.image
my_map.add_label_to_image("middle", Coordinate((33.4279533+33.4618937)/2.0, (-112.1082946 + -112.0371188)/2.0), False, 1.0)

im.show()

