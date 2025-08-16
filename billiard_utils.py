"""
Utilities for computing the dynamics of an ensemble of particles inside a triangular box.

propogate_list() computes the ensemble dynamics.
"""

import numpy as np


class Triangle:
    """
    Specifies triangle shape and other properties
    Vertex coordinates are (L1r,0), (L1l,0), (0,L2)
       /|\    |
      / | \   L2
     /__|__\  |
     L1l L1R

    Computes left_side_vect, the normalized vector that points from the
    bottom left corner to the top corner.

    Computes right_side_vect, the normalized vector that points from
    the top corner to the bottom right corner.
    """
    def __init__(self, L1l, L1r, L2):
        self.L1l = L1l
        self.L1r = L1r
        self.L2 = L2
        self.L_array = np.array([L1l, L1r, L2])
        self.left_side_vect = np.array([1, -L2/L1l]) / np.sqrt(1
                                                               + L2**2/L1l**2)
        self.right_side_vect = np.array([1, -L2/L1r]) / np.sqrt(1
                                                                + L2**2/L1r**2)
        self.Area = 0.5*(L1r-L1l)*L2


t_list = np.zeros(3)  # Initializes array to be used in next_collision
def next_collision(x, y, vx, vy, max_t, triangle, epsilon=1e-8):
    """
    Computes the state of particle immediate following the next
    collision with a wall, UNLESS no collision occurs within time
    max_t, in which case it returns that state of the particle
    after time max_t.

    Parameters:
        x (float): initial x of particle
        y (float): initial y of particle
        vx (float): initial vx of particle
        vy (float): initial vy of particle
        max_t: Maximum time
        triangle (Triangle): Triangle object that defines the walls
        epsilon (float): small parameter used to ensure particle
                         remains inside triangle after collision.

    Returns:
        x: final x after collision
        y: final y after collision
        vx: final vx after collision
        vy: final vy after collision
        collision_time: time elapsed until collision OR max_t if no
                        collision occurs.
        collision_occured: TRUE if collision occurs, False otherwise.
    """

    def v_components(vx, vy, side_vect):
        """
        Computes velocity components parallel to and perpendicular
        to colliding wall

        Parameters:
            vx, vy: velocity components
            side_vect: vector in direction of side. Typically,
                side_vect = triangle.left_side_vect or
                side_vect = triangle.right_side_vect
        """
        v_parallel = side_vect * (vx*side_vect[0] + vy*side_vect[1])
        v_perpendicular = [vx - v_parallel[0], vy - v_parallel[1]]
        return v_parallel, v_perpendicular

    # t_left, t_bottom, and t_right are the times at which the
    # particle will collide with the left, bottom, and right wall,
    # (assuming it does not hit another wall first!)
    # These times may be negative if the particle would collide
    # with the wall in the past. The smallest positive value
    # indicates which collision will occur next.
    t_left = (
        (triangle.L2 - triangle.L2*x/triangle.L1l - y)
        / (vy + triangle.L2*vx/triangle.L1l)
    )
    t_bottom = -y/vy
    t_right = (
        (triangle.L2 - triangle.L2*x/triangle.L1r - y)
        / (vy + triangle.L2*vx/triangle.L1r)
    )

    # List of times of wall collisions
    t_list[0], t_list[1], t_list[2] = t_left, t_bottom, t_right

    # Set negative times to inf (because collision doesn't occur)
    t_list[t_list <= 0] = np.inf

    if np.all(t_list > max_t):
        # No collision occurs before time max_t
        # Return particle state at time max_t, collision_occurs=False
        return x+vx*max_t, y+vy*max_t, vx, vy, max_t, False

    else:
        # A collision does occur before max_t.
        # Determine which wall particle collides with.
        # collision_wall = 0 : left wall
        # collision_wall = 1 : bottom wall
        # collision_wall = 2 : right wall
        # collision_time = time until collision.
        collision_wall = np.argmin(t_list)
        collision_time = t_list[collision_wall]

        if collision_wall == 0:  # Left wall collision
            x += vx*collision_time
            y = triangle.L2*(1.-x/triangle.L1l) - epsilon
            v_parallel, v_perpend = v_components(vx, vy,
                                                 triangle.left_side_vect)
            vx = v_parallel[0] - v_perpend[0]
            vy = v_parallel[1] - v_perpend[1]
        elif collision_wall == 1:  # Bottom wall collision
            y = 0.
            x += vx*collision_time
            vy *= -1
        else:  # Right wall collision
            x += vx*collision_time
            y = triangle.L2*(1.-x/triangle.L1r) - epsilon
            v_parallel, v_perpend = v_components(vx, vy,
                                                 triangle.right_side_vect)
            vx = v_parallel[0] - v_perpend[0]
            vy = v_parallel[1] - v_perpend[1]
        return x, y, vx, vy, collision_time, True


def trajectory_collisions(x, y, vx, vy, t, triangle, max_collisions=10000):
    """
    Generates a trajectory where the position and time of each
    collision are recorded.

    Parameters:
        x: initial x of particle
        y: initial y of particle
        vx: initial vx of particle
        vy: initial vy of particle
        t: trajectory time
        triangle: Triangle object that defines the walls

    Returns:
        xs: x components of trajectory.
        ys: y components of trajectory.
            xs[0] = x, ys[0] = y (the initial conditions)
            xs[-1],ys[-1] is the particle position after
            time t, which is NOT a time at which a collision
            occurs (in general).
        ts: times of each collision and final point. ts[0] = 0.
    """

    xs, ys, ts = [x], [y], [0]  # Initialize lists with initial conditions.

    for i in range(max_collisions):
        # Find updated state after next collision OR after the remaining
        # time (t-ts[-1]) runs out.
        x, y, vx, vy, collision_time, collision_occured = \
            next_collision(xs[-1], ys[-1], vx, vy, t-ts[-1], triangle)

        xs.append(x)
        ys.append(y)

        if collision_occured:
            ts.append(ts[-1] + collision_time)
        else:
            ts.append(t)  # Append final time
            break
    else:
        # If loop completes without breaking, then max_collisions
        # was reached without simulating the full time t.
        raise RuntimeError("max_collisions reached before time t.")

    return xs, ys, ts


def propogate(x, y, vx, vy, t, triangle):
    """
    Propogates a particle forward by time t

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


    This is a recursive function which steps forward collision-after
    -collision using the next_collision function until the particle
    has been propogated forward by time t.

    """

    # Computes updated state after next collision OR after time t
    # if no collision occurs
    x, y, vx, vy, collision_time, collision_occured = next_collision(
        x, y, vx, vy, t, triangle)

    if collision_occured:
        # If collision occurs, there is more time left to be simulated!
        # Thus, continue propogating
        return propogate(x, y, vx, vy, t-collision_time, triangle)

    else:
        # No collision occurs, so the final state at time t is reached.
        return x, y, vx, vy


def propogate_list(x_list, y_list, vx_list, vy_list, t, triangle):
    """
    Propogate a list of particles

    Parameters:
        x_list: list of x values
        Y_list: list of Y values
        Vx_list: list of Vx values
        VY_list: list of VY values

    Returns:
        tuple (x_list,y_list,vx_list,vy_list) with updated states.
    """
    for i in range(len(x_list)):
        x_list[i], y_list[i], vx_list[i], vy_list[i] = \
                propogate(x_list[i], y_list[i], vx_list[i], vy_list[i],
                          t, triangle)

    return x_list, y_list, vx_list, vy_list
