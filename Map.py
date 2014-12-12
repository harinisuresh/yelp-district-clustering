"""Map class for hanlding coordinate conversion and adding labels to map image."""
from MapUtils import Position, Coordinate
from PIL import Image
from PIL import ImageFont, ImageDraw, ImageOps

class Map:
    def __init__(self, top_left_coord, top_right_coord, bottom_left_coord, bottom_right_coord, image_path):
        self.top_left_coord = top_left_coord
        self.top_right_coord = top_right_coord
        self.bottom_left_coord = bottom_left_coord
        self.bottom_right_coord = bottom_right_coord
        self.image =  Image.open(image_path)
        self.image_path = image_path

    def real_width(self):
        return abs(self.top_left_coord.longitude - self.top_right_coord.longitude)

    def real_height(self):
        return abs(self.top_left_coord.latitude - self.bottom_left_coord.latitude)

    def image_width(self):
        return self.image.size[0]

    def image_height(self):
        return self.image.size[1]

    def world_coordinate_to_image_position(self, coordinate, from_bottom_left=False):
        x_proportion = (coordinate.longitude - self.top_left_coord.longitude)/self.real_width()
        y_proportion = (abs(self.top_left_coord.latitude) - abs(coordinate.latitude))/self.real_height()
        x = x_proportion * self.image_width()
        y = y_proportion * self.image_height()
        if from_bottom_left:
            y = self.image_height() - y
        return Position(x,y)

        """Accepts either coordinate or position."""
    def add_label_to_image(self, label_text, position=None, coordinate=None, rotated=False, weight = 1.0):
        if coordinate == None and position == None:
            raise Exception("Coordinate and position can't both be None")
        if position == None:
            position = self.world_coordinate_to_image_position(coordinate)
        draw_txt = ImageDraw.Draw(self.image)
        font_size = int(weight*16.0)
        position = Map.center_label_pos(position, font_size, label_text, rotated)
        font = ImageFont.truetype("fonts/BEBAS___.TTF", font_size)
        draw_txt.text((position.x, position.y), label_text, font=font, fill=(0, 0, 0, 255))

    @staticmethod
    def center_label_pos(img_pos, font_size, label_text, rotated=False):
        pos = Position(img_pos.x, img_pos.y)
        if rotated:
            pos.y -= (font_size * len(label_text))/4.0
            pos.x -= (font_size)/4.0
        else:
            pos.x -= (font_size * len(label_text))/4.0
            pos.y -= (font_size)/4.0
        return pos

    @staticmethod
    def pheonix():
        """Returns the map object of phoenix"""
        imagePath = "images/phoenix.png"
        top_latitude = 33.465830
        bottom_latitude = 33.4209533
        left_longitude = -112.1172946
        right_longitude = -112.0071188
        return Map(Coordinate(top_latitude,left_longitude), \
         Coordinate(top_latitude,right_longitude), Coordinate(bottom_latitude,left_longitude),\
         Coordinate(bottom_latitude,right_longitude), imagePath)

    @staticmethod
    def vegas():
        """Returns the map object of vegas"""
        imagePath = "images/vegas2.png"
        top_latitude = 36.310359
        bottom_latitude = 35.980255
        left_longitude = -115.357904
        right_longitude = -114.949179

        return Map(Coordinate(top_latitude,left_longitude), \
         Coordinate(top_latitude,right_longitude), Coordinate(bottom_latitude,left_longitude),\
         Coordinate(bottom_latitude,right_longitude), imagePath)
