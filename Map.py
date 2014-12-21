"""Module for Map Class"""

from MapUtils import Position, Coordinate
from PIL import Image
from PIL import ImageFont, ImageDraw, ImageOps

class Map:
    """Map class for hanlding coordinate conversion and adding labels to map image."""
    def __init__(self, top_left_coord, top_right_coord, bottom_left_coord, bottom_right_coord, image_path):
        """
        Constructor for Map Object

        Parameters:
        top_left_coord - Coordinate Object representing the top left corner
        top_right_coord - Coordinate Object representing the top right corner
        bottom_left_coord - Coordinate Object representing the bottom left corner
        bottom_right_coord - Coordinate Object representing the bottom right corner
        image_path - path of the image of the map

        Returns:
            A Map object.
        """
        self.top_left_coord = top_left_coord
        self.top_right_coord = top_right_coord
        self.bottom_left_coord = bottom_left_coord
        self.bottom_right_coord = bottom_right_coord
        self.image =  Image.open(image_path)
        self.image_path = image_path

    def real_width(self):
        """
        Get the coordinate width of this map.

        Returns:
            The coordinate width of this map.
        """
        return abs(self.top_left_coord.longitude - self.top_right_coord.longitude)

    def real_height(self):
        """
        Get the coordinate height of this map.

        Returns:
            The coordinate height of this map.
        """
        return abs(self.top_left_coord.latitude - self.bottom_left_coord.latitude)

    def image_width(self):
        """
        Get the width of this map's image.

        Returns:
            The width of this map's image.
        """
        return self.image.size[0]

    def image_height(self):
        """
        Get the height of this map's image.

        Returns:
            The height of this map's image.
        """
        return self.image.size[1]

    def world_coordinate_to_image_position(self, coordinate, from_bottom_left=False):
        """
        Get the image position corresponding to the world coordinate.

        Parameters:
        coordinate - Coordinate Object to be transformed
        from_bottom_left - Determines if the new coordinate system should be 0 at the bottom left or top left.

        Returns:
            The image position corresponding to the world coordinate as a Position object.
        """
        x_proportion = (coordinate.longitude - self.top_left_coord.longitude)/self.real_width()
        y_proportion = (abs(self.top_left_coord.latitude) - abs(coordinate.latitude))/self.real_height()
        x = x_proportion * self.image_width()
        y = y_proportion * self.image_height()
        if from_bottom_left:
            y = self.image_height() - y
        return Position(x,y)

    def add_label_to_image(self, label_text, position=None, coordinate=None, rotated=False, weight = 1.0):
        """
        Add a label to the image.

        Parameters:
        label_text - The text of the label
        position - the image position for the label
        coordinate - the coordinate position for the label
        rotated - If the label should be rotated
        weight - Size of the label

        Returns:
            None
        """
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
        """
        Get the center of a label from its text and font size.

        Parameters:
        img_pos - The left-aligned position of the label.
        font_size - The font size of the label
        label_text - The text of the label
        rotated - If the label is rotated
        Returns:
            The center of a label.
        """
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
        top_latitude = 33.788493
        bottom_latitude = 33.129717
        left_longitude = -112.412109
        right_longitude = -111.622467
        return Map(Coordinate(top_latitude,left_longitude), \
         Coordinate(top_latitude,right_longitude), Coordinate(bottom_latitude,left_longitude),\
         Coordinate(bottom_latitude,right_longitude), imagePath)

    @staticmethod
    def vegas():
        """Returns the map object of vegas"""
        imagePath = "images/vegas.png"
        top_latitude = 36.310359
        bottom_latitude = 35.980255
        left_longitude = -115.357904
        right_longitude = -114.949179
        return Map(Coordinate(top_latitude,left_longitude), \
         Coordinate(top_latitude,right_longitude), Coordinate(bottom_latitude,left_longitude),\
         Coordinate(bottom_latitude,right_longitude), imagePath)

    @staticmethod
    def waterloo():
        """Returns the map object of waterloo"""
        imagePath = "images/waterloo.png"
        top_latitude = 43.544364
        bottom_latitude = 43.410581
        left_longitude = -80.635872
        right_longitude = -80.451164
        return Map(Coordinate(top_latitude,left_longitude), \
         Coordinate(top_latitude,right_longitude), Coordinate(bottom_latitude,left_longitude),\
         Coordinate(bottom_latitude,right_longitude), imagePath)

    @staticmethod
    def edinburgh():
        """Returns the map object of edinburgh"""
        imagePath = "images/edinburgh.png"
        top_latitude = 55.986704
        bottom_latitude = 55.907306
        left_longitude = -3.251266
        right_longitude = -3.121834

        return Map(Coordinate(top_latitude,left_longitude), \
         Coordinate(top_latitude,right_longitude), Coordinate(bottom_latitude,left_longitude),\
         Coordinate(bottom_latitude,right_longitude), imagePath)

    @staticmethod
    def madison():
        """Returns the map object of madison"""
        imagePath = "images/madison.png"
        top_latitude = 43.215279
        bottom_latitude = 42.960537
        left_longitude = -89.573593
        right_longitude = -89.223404

        return Map(Coordinate(top_latitude,left_longitude), \
         Coordinate(top_latitude,right_longitude), Coordinate(bottom_latitude,left_longitude),\
         Coordinate(bottom_latitude,right_longitude), imagePath)
