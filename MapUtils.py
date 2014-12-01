"""Utils used by map"""

import colorsys

class Position:
    """Position is a class for storing x and y"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Position: (x, y)" + str((self.x,self.y))
class Coordinate:
    """Coordinate is a class for storing lat and long"""
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def __str__(self):
        return "Coordinate (lat, long): " + str((self.latitude,self.longitude))

class Rectangle:
    def __init__(self, pt1, pt2):
        """Initialize a rectangle from two points."""
        self.set_points(pt1, pt2)

    def contains(self, point):
        """Return true if a point is inside the rectangle."""
        x = point.x
        y = point.y
        return (self.left <= x <= self.right and
                  self.top <= y <= self.bottom)
      
    def set_points(self, pt1, pt2):
        """Reset the rectangle coordinates."""
        (x1, y1) = pt1.x, pt1.y
        (x2, y2) = pt2.x, pt2.y
        self.left = min(x1, x2)
        self.top = min(y1, y2)
        self.right = max(x1, x2)
        self.bottom = max(y1, y2)

    def center(self):
        return Position((self.left + self.right)*0.5 , (self.top + self.bottom)*0.5)

def create_n_unique_colors(n):
    colors = []
    h = 0.0
    dh = 1.0/n
    for i in range(n):
        l = .5
        s = .8
        h += dh
        rgb_color = list(colorsys.hls_to_rgb(h, l, s))
        colors.append(rgb_color)
    return colors
