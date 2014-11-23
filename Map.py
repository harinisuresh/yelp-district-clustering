from MapUtils import Position, Coordinate
from PIL import Image
from PIL import ImageFont, ImageDraw, ImageOps

class Map:
    def __init__(self, top_left_coord, top_right_coord, bottom_left_coord, bottom_right_coord, image):
        self.top_left_coord = top_left_coord
        self.top_right_coord = top_right_coord
        self.bottom_left_coord = bottom_left_coord
        self.bottom_right_coord = bottom_right_coord
        self.image = image

    def real_width(self):
        return abs(self.top_left_coord.longitude - self.top_right_coord.longitude)

    def real_height(self):
        return abs(self.top_left_coord.latitude - self.bottom_left_coord.latitude)

    def image_width(self):
        return self.image.size[0]

    def image_height(self):
        return self.image.size[1]

    def world_coordinate_to_image_position(self, coordinate):
        x_proportion =  abs(coordinate.longitude - self.top_left_coord.longitude)/self.real_width()
        y_proportion =  abs(coordinate.latitude - self.top_left_coord.latitude)/self.real_height()
        x = x_proportion * self.image_width()
        y = y_proportion * self.image_height()
        print  "(x,y)", (x,y)
        print  "height", self.real_height()
        print  "y_proportion", y_proportion
        print "coordinate.latitude", coordinate.latitude
        print "(coordinate.latitude - self.top_left_coord.latitude)", (coordinate.latitude - self.top_left_coord.latitude)

        return Position(x,y)

    def add_label_to_image(self, label_text, coordinate, rotated=False, weight = 1.0):
        img_pos = self.world_coordinate_to_image_position(coordinate)
        draw_txt = ImageDraw.Draw(self.image)
        font_size = int(weight*20.0)
        img_pos = Map.center_label_pos(img_pos, font_size, label_text, rotated)
        font = ImageFont.truetype("fonts/ProximaNova.ttf", font_size)
        draw_txt.text((img_pos.x, img_pos.y), label_text, font=font, fill=(0, 0, 0, 255))

    @staticmethod
    def center_label_pos(img_pos, font_size, label_text, rotated=False):
        pos = Position(img_pos.x, img_pos.y)
        if rotated:
            pos.y -= (font_size * len(label_text))/4.0
            pos.x -= (font_size * len(label_text))/8.0
        else:
            pos.x -= (font_size * len(label_text))/4.0
            pos.y -= (font_size * len(label_text))/8.0
        return pos

    @staticmethod
    def pheonix():
        """Returns the map object of phoenix"""
        image = Image.open("images/phoenix.png")
        top_latitude = 33.4618937
        botton_latitude = 33.4279533
        left_longitude = -112.1082946
        right_longitude = -112.0371188
        return Map(Coordinate(top_latitude,left_longitude), \
         Coordinate(top_latitude,right_longitude), Coordinate(botton_latitude,left_longitude),\
         Coordinate(botton_latitude,right_longitude), image)
