import numpy as np

class Triangle:
    """ Specifies triangle shape and other properties
    Vertex coordinates are (L1r,0), (L1l,0), (0,L2)
       /|\    |
      / | \   L2
     /__|__\  |
     L1l L1R
    """
    def __init__(self,L1l,L1r,L2):
        self.L1l = L1l
        self.L1r = L1r
        self.L2 = L2
        self.left_side_vect = np.array([1,-L2/L1l]) / np.sqrt(1 + L2**2/L1l**2)
        self.right_side_vect = np.array([1,-L2/L1r]) / np.sqrt(1 + L2**2/L1r**2)


t_list = np.zeros(3) # Initializes array to be used in next_collision
def next_collision(x,y,vx,vy,max_t,triangle):
    """ Computes the state of particle immediate following the next collision
        with a wall, UNLESS no collision occurs within time max_t, in which
        case it returns that state of the particle after time max_t.

        Parameters:
            x: initial x of particle
            y: initial y of particle
            vx: initial vx of particle
            vy: initial vy of particle
            max_t: Maximum time
            triangle: Triangle object that defines the walls

        Returns:
            x: final x after collision
            y: final y after collision
            vx: final vx after collision
            vy: final vy after collision
            collision_time: time elapsed until collision OR max_t if no 
                            collision occurs.
            collision_occured: TRUE if collision occurs, False otherwise.
            

    """
    t_left = (triangle.L2 - triangle.L2*x/triangle.L1l - y) / (vy + triangle.L2*vx/triangle.L1l)
    t_bottom = -y/vy
    t_right = (triangle.L2 - triangle.L2*x/triangle.L1r - y) / (vy + triangle.L2*vx/triangle.L1r)

    t_list[0],t_list[1],t_list[2] = t_left,t_bottom,t_right
    t_list[t_list<=0] = np.inf

    if np.all(t_list>max_t): # No collision
        return x+vx*max_t, y+vy*max_t, vx, vy, max_t, False
    else:
        collision_wall = np.argmin(t_list)
        collision_time = t_list[collision_wall]
        if collision_wall==0: #left
            x += vx*collision_time
            y = triangle.L2*(1.-x/triangle.L1l) - 0.00000001
            v_parallel = triangle.left_side_vect * (vx*triangle.left_side_vect[0] + vy*triangle.left_side_vect[1])
            v_perpendicular = [vx - v_parallel[0], vy - v_parallel[1]]
            vx = v_parallel[0] - v_perpendicular[0]
            vy = v_parallel[1] - v_perpendicular[1]
        elif collision_wall==1: #bottom
            y = 0.
            x += vx*collision_time
            vy *= -1
        else: #hypotenuse
            x += vx*collision_time
            y = triangle.L2*(1.-x/triangle.L1r) - 0.00000001
            v_parallel = triangle.right_side_vect * (vx*triangle.right_side_vect[0] + vy*triangle.right_side_vect[1])
            v_perpendicular = [vx - v_parallel[0], vy - v_parallel[1]]
            vx = v_parallel[0] - v_perpendicular[0]
            vy = v_parallel[1] - v_perpendicular[1]
        return x,y,vx,vy,collision_time, True


def trajectory_collisions(x,y,vx,vy,t,triangle):
    """ Generates a trajectory where the position and time of each collision
        are recorded.

        Parameters:
            x: initial x of particle
            y: initial y of particle
            vx: initial vx of particle
            vy: initial vy of particle
            t: trajectory time
            triangle: Triangle object that defines the walls

        Returns:
            xs: x components of trajectory. xs[0] = x (initial condition) and
                xs[-1] is the particle position after time t, which is NOT
                the time at which a collision occurs (in general).
            ys: y components of trajectory. ys[0] = y (initial condition) and
                ys[-1] is the particle position after time t, which is NOT
                the time at which a collision occurs (in general).
            ts: times of each collision and final point. ts[0] = 0.

    """

    xs = [x]
    ys = [y]
    ts = [0]
    for i in range(10000):
        x,y,vx,vy,collision_time, collision_occured = next_collision(xs[-1],ys[-1],vx,vy,t-ts[-1],triangle)
        xs += [x]
        ys += [y]
        if collision_occured:
            ts += [ts[-1]+collision_time]
        else:
            ts += [t] # Append final time
            break
    else:
        print('reached end of loop without finishing.')
        quit()

    return xs,ys,ts


def propogate(x,y,vx,vy,t,triangle):
    """ Propogates a particle forward by time t

        Parameters:
            x: initial x of particle
            y: initial y of particle
            vx: initial vx of particle
            vy: initial vy of particle
            t: trajectory time
            triangle: Triangle object that defines the walls

        Returns:
            x: final x
            y: final y
            vx: final vx     
            vy: final vy     


        This is a recursive function which steps forward collision-after-collision
        using the next_collision function until the particle has been propogated
        forward by time t.

    """
    x,y,vx,vy,collision_time,collision_occured  = next_collision(x,y,vx,vy,t,triangle)
    if collision_occured:
        return propogate(x,y,vx,vy,t-collision_time,triangle)
    else:
        return x,y,vx,vy


#