import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_with_arrow(ax,x,y,color='k',linestyle='-',arrow_style='solid',
                    tail_style='o',length=0.05,angle=0.4):
    """ Geneterates x vs y line plot (pyplot.plot) terminating in an arrow.
        
        Paramters:
            ax: pyplot axes on which to plot
            x: x coordinates
            y: y coordinates
            color: line color
            linestyle: pyplot linestyle
            arrow_style ('solid' or 'V'): solid arrow or V-shaped arrow
            tail_style (None,'o','s',...): style of initial point
            length: arrow length
            angle: angle of arrow tip (adjusts width of arrow head)
    """
    
    # Plot line
    ax.plot(x,y,linestyle=linestyle,color=color)
    
    # Plot "tail" (initial point)
    ax.plot(x[0],y[0],marker=tail_style,color=color,markersize=7)

    # Defining the direction of the arrow
    v = np.array([x[-2] - x[-1],y[-2]-y[-1]]) #direction
    v *= length / np.linalg.norm(v) #normalized direction

    # Rotation matrix
    R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    
    # Sides of arrow
    side1 = R@v
    side2 = R.T@v

    if arrow_style=='V': # V-shaped arrow (plots sides only)
        ax.plot([x[-1],x[-1]+side1[0]],[y[-1],y[-1]+side1[1]],color=color)
        ax.plot([x[-1],x[-1]+side2[0]],[y[-1],y[-1]+side2[1]],color=color)
    
    elif arrow_style=='solid': # Solid arrow (plots triangular patch)
        vertices = [(x[-1],y[-1]),(x[-1]+side1[0],y[-1]+side1[1]),(x[-1]+side2[0],y[-1]+side2[1])]
        triangle = patches.Polygon(vertices, closed=True, color='black')
        ax.add_patch(triangle)

