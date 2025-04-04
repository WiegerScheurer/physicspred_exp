import os
# import OmegaConf
from psychopy.visual import Circle, ShapeStim, Rect, ImageStim
from .utils import get_bounce_dist

# # Load configuration from YAML file
# config_path = os.path.join(os.path.dirname(__file__), os.pardir, "config_lumin.yaml")
# config = OmegaConf.load(config_path)

def create_circle_fixation(win, radius=0.1, color=(1, 1, 1),
                           edges=100, **kwargs):
    """ Creates a circle fixation dot with sensible defaults. """
    return Circle(win, radius=radius, color=color, edges=edges, **kwargs)


# Draw the fixation cross by drawing both lines
def create_cross_fixation(win, length=30, thickness=8,
                  color="#33CC00", **kwargs):
    """ Draws a fixation cross on the screen. """
    
        
    # Create the horizontal line of the cross
    horizontal_line = ShapeStim(
        win,
        vertices=[(-length / 2, 0), (length / 2, 0)],
        lineWidth=thickness,
        closeShape=False,
        lineColor=color,
        **kwargs
    )

    # Create the vertical line of the cross
    vertical_line = ShapeStim(
        win,
        vertices=[(0, -length / 2), (0, length / 2)],
        lineWidth=thickness,
        closeShape=False,
        lineColor=color,
        **kwargs
    )

    # horizontal_line.draw()
    # vertical_line.draw()
    return horizontal_line, vertical_line

def create_occluder(win, radius=360, opacity=1, color="#764A15", **kwargs):
    """ Creates an occluder. """
    
        
    # Add opacity = .5 to make see-through
    return Rect(win, width=radius, height=radius, fillColor=color, lineColor=color, 
                pos=(0, 0), opacity=opacity, interpolate=True, **kwargs)
    

def create_interactor(win, trial, ball_radius, height, width, path_45, path_135, **kwargs):
    """ Creates an interactor. """
    bounce_dist = get_bounce_dist(ball_radius + (width / 2 * 1.8)) # 1.8 factor is due to the that now we use an image

            
    line_45_bottom = ImageStim(
        win,
        # image="/Users/wiegerscheurer/Stimulus_material/45_flat_beige.png", 
        image=path_45,
        size=(height, height),
        pos=(bounce_dist, -(bounce_dist)),
        opacity=1,
        interpolate=True,
    )

    line_45_top = ImageStim(
        win,
        # image="/Users/wiegerscheurer/Stimulus_material/45_flat_white.png", 
        image=path_45,
        size=(height, height),
        pos= (-(bounce_dist), bounce_dist),
        opacity=1,
        interpolate=True,
    )

    line_135_bottom = ImageStim(
        win,
        # image="/Users/wiegerscheurer/Stimulus_material/135_flat_white.png", 
        image=path_135,
        size=(height, height),
        pos=(-bounce_dist, -(bounce_dist)),
        opacity=1,
        interpolate=True,
    )

    line_135_top = ImageStim(
        win,
        # image="/Users/wiegerscheurer/Stimulus_material/135_flat_white.png", 
        image=path_135,
        size=(height, height),
        pos= ((bounce_dist), bounce_dist),
        opacity=1,
        interpolate=True,
    )
    
    # Draw interactor line if applicable
    if trial:
        if trial[:-2] == "45_top":
            # line_135_top.draw()
            return line_135_top
        elif trial[:-2] == "45_bottom":
            # line_135_bottom.draw()
            return line_135_bottom
        elif trial[:-2] == "135_top":
            # line_45_top.draw()
            return line_45_top
        elif trial[:-2] == "135_bottom":
            # line_45_bottom.draw()
            return line_45_bottom
    
# Helper function to draw screen borders and other elements
def create_screen_borders(win, dims, square_size, **kwargs):    # Create the grey borders
    left_border = Rect(
        win=win,
        width=(dims[0] - square_size) / 2,
        height=dims[1],
        fillColor="black",
        lineColor="black",
        pos=[-(dims[0] - square_size) / 4 - square_size / 2, 0],
    )

    right_border = Rect(
        win=win,
        width=(dims[0] - square_size) / 2,
        height=dims[1],
        fillColor="black",
        lineColor="black",
        pos=[(dims[0] - square_size) / 4 + square_size / 2, 0],
    )

    top_border = Rect(
        win=win,
        width=dims[0],
        height=(dims[1] - square_size) / 2,
        fillColor="black",
        lineColor="black",
        pos=[0, (dims[1] - square_size) / 4 + square_size / 2],
    )

    bottom_border = Rect(
        win=win,
        width=dims[0],
        height=(dims[1] - square_size) / 2,
        fillColor="black",
        lineColor="black",
        pos=[0, -(dims[1] - square_size) / 4 - square_size / 2],
    )

    return left_border, right_border, top_border, bottom_border
    # left_border.draw()
    # right_border.draw()
    # top_border.draw()
    # bottom_border.draw()
    

        
    # fixation.draw()
    # draw_fixation()