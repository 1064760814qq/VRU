import time

# from shapely.geometry import Point
# from shapely.geometry import Polygon
import pyproj
# import csv


def time_stamp(time_s):
    time_trans = time.localtime(int(time_s))
    return time_trans.tm_mon, time_trans.tm_mday, time_trans.tm_hour

# print(time_stamp(1545646))

def proj_trans(lon, lat):
    p1 = pyproj.Proj(init="epsg:4326")
    p2 = pyproj.Proj(init="epsg:32650")
    x1, y1 = p1(lon, lat)
    x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
    return x2, y2

print(proj_trans(39.89944,116.47236))