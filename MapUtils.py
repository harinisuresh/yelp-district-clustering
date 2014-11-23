"""Utils used by map"""

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