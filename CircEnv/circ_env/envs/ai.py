import numpy as np
import circ_env.envs.hockey_world as const

class AI(object):
    def __init__(self, mallet, puck, mode, dim):
        self.mallet = mallet
        self.puck = puck
        self.mode = mode
        self.dim = dim
        self.vel = 5.0
        self._vel = [0,0]

    def move(self):
        
        # 0,0 levo gor
        px, py = self.mallet.position
        vx, vy = self.mallet.linearVelocity
        px *= const.PPM
        py *= const.PPM
        py = self.dim.height - py
        vx *= const.PPM
        vy *= -const.PPM
        
        puck = self.puck
        puck_px, puck_py = self.puck.position
        puck_vx, puck_vy = self.puck.linearVelocity
        
        puck_px *= const.PPM 
        puck_py *= const.PPM
        puck_py = self.dim.height - puck_py
        puck_vx *= const.PPM
        puck_vy *= -const.PPM
        
        if self.mode == 'top':
            goal_px, goal_py = (self.dim.center[0], self.dim.rink_top - 10)
        elif self.mode == 'bottom':
            goal_px, goal_py = (self.dim.center[0], self.dim.rink_bottom + 10)

        if self.mode == 'top':
            reachable = self.dim.rink_top <= puck_py <=  self.dim.center[1]
        elif self.mode == 'bottom':
            reachable = self.dim.center[1] <= puck_py <=  self.dim.rink_bottom

        x, y = 0, 0
        if not reachable:
            if self.mode == 'top':
                target_px, target_py = (self.dim.center[0], self.dim.rink_top + 40)
            elif self.mode == 'bottom':
                target_px, target_py = (self.dim.center[0], self.dim.rink_bottom - 40)
            def defend_goal(goal, p):
                diff = goal - p
                if abs(diff) < 5: return  0
                elif diff > 0:    return  self.vel
                else:             return -self.vel
            x = defend_goal(target_px, px)
            y = defend_goal(target_py, py)
            # print('{:15} {:4d} {:4d}'.format('not reachable', x, y))

        else:
            #if puck_vy <= 0:
            if puck_px < px: x = -self.vel
            if puck_px > px: x = self.vel
            if puck_py < py: y = -self.vel
            if puck_py > py: y = self.vel
        
        self._vel = [x, -y]
        
        return self._vel
    