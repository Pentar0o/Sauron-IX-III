from collections import namedtuple

Vector = namedtuple('Vector', ['x', 'y', 'direction_x', 'direction_y'])
Point = namedtuple('Point', ['x', 'y'])
Rect = namedtuple('Rect', ['xmin', 'ymin', 'xmax', 'ymax'])
Config = namedtuple('Config', ['access_key', 'secret_key', 'bucket_name'])
