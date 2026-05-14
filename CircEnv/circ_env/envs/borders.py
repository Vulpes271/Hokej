import numpy as np
from circ_env.envs.line import Line
from circ_env.envs.dimensions import Dimensions
from Box2D import *
import circ_env.envs.hockey_world as C
from operator import itemgetter

dim = Dimensions()

top_wall          = Line(dim.arc_top_left_start, dim.arc_top_right_end)
bottom_wall       = Line(dim.arc_bottom_left_end, dim.arc_bottom_right_start)
left_wall         = Line(dim.arc_top_left_end, dim.arc_bottom_left_start)
right_wall        = Line(dim.arc_top_right_start, dim.arc_bottom_right_end)

top_left_wall     = Line(dim.arc_top_left_start, dim.post_top_left)
top_right_wall    = Line(dim.post_top_right, dim.arc_top_right_end)
bottom_left_wall  = Line(dim.arc_bottom_left_end, dim.post_bottom_left)
bottom_right_wall = Line(dim.post_bottom_right, dim.arc_bottom_right_start)

center_line       = Line(dim.center_left, dim.center_right)

# Add Corners
top_left_corner     = Line.generate_bezier_curve(dim.arc_top_left, dim.bezier_ratio)
top_right_corner    = Line.generate_bezier_curve(dim.arc_top_right, dim.bezier_ratio)
bottom_left_corner  = Line.generate_bezier_curve(dim.arc_bottom_left, dim.bezier_ratio)
bottom_right_corner = Line.generate_bezier_curve(dim.arc_bottom_right, dim.bezier_ratio)

borders = [top_left_wall, top_right_wall, bottom_left_wall, bottom_right_wall, left_wall, right_wall] + \
                top_left_corner + top_right_corner + bottom_left_corner + bottom_right_corner
                
                
def generateBorder(border):
    #border is list of Line objects
    #returns list of points
            vertices = []
            for line in border:
                vertices.append(line.p2)
                vertices.append(line.p1)
            vert2 = []
            for arr in vertices:
                a = float(arr[0])
                b = float(dim.height-arr[1])
                vert2.append((a/C.PPM,b/C.PPM))
            
            vert2 = np.array(vert2)   
            #vert2 = np.unique(vert2, axis=1)
            #vert2 = np.sort(vert2)
            vert2 = vert2.tolist()
            vert2 = sorted(vert2, key=itemgetter(1))
            #vert2 = vert2.reverse()
        
            shapes = []
            p1 = None
            p2 = None

            for i,line in enumerate(vert2): 
                p2 = line
                if i != 0:
                    shapes.append(b2EdgeShape(vertices=[p1, p2]))
                p1 = p2
                
            return shapes