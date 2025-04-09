import os
# import OmegaConf
from psychopy.visual import Circle, ShapeStim, Rect, ImageStim
from utils import get_bounce_dist

# # Load configuration from YAML file
# config_path = os.path.join(os.path.dirname(__file__), os.pardir, "config_lumin.yaml")
# config = OmegaConf.load(config_path)

def create_ball(win, ball_radius=0.1, **kwargs):
    """ Creates a ball with sensible defaults. """
    # config = components["config"]
    # ball_radius = config["ball"]["radius"]
    
    # Create the ball
    return Circle(win, 
            radius=ball_radius, 
            edges=64,
            fillColor="white",#config["ball_fillcolor"], 
            lineColor="white", #config["ball_linecolor"], 
            interpolate=True,
            opacity=1)

# ball = Circle(win, 
#             radius=ball_radius, 
#             edges=64,
#             fillColor="white",#config["ball_fillcolor"], 
#             lineColor="white", #config["ball_linecolor"], 
#             interpolate=True,
#             opacity=1)


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

            
    # line_45_bottom = ImageStim(
    #     win,
    #     # image="/Users/wiegerscheurer/Stimulus_material/45_flat_beige.png", 
    #     image=path_45,
    #     size=(height, height),
    #     pos=(bounce_dist, -(bounce_dist)),
    #     opacity=1,
    #     interpolate=True,
    # )

    # line_45_top = ImageStim(
    #     win,
    #     # image="/Users/wiegerscheurer/Stimulus_material/45_flat_white.png", 
    #     image=path_45,
    #     size=(height, height),
    #     pos= (-(bounce_dist), bounce_dist),
    #     opacity=1,
    #     interpolate=True,
    # )

    # line_135_bottom = ImageStim(
    #     win,
    #     # image="/Users/wiegerscheurer/Stimulus_material/135_flat_white.png", 
    #     image=path_135,
    #     size=(height, height),
    #     pos=(-bounce_dist, -(bounce_dist)),
    #     opacity=1,
    #     interpolate=True,
    # )

    # line_135_top = ImageStim(
    #     win,
    #     # image="/Users/wiegerscheurer/Stimulus_material/135_flat_white.png", 
    #     image=path_135,
    #     size=(height, height),
    #     pos= ((bounce_dist), bounce_dist),
    #     opacity=1,
    #     interpolate=True,
    # )
    line_45_bottom = Rect(win, width=100, height=500, fillColor="red", lineColor="green", 
                pos=(0, 0), opacity=1, interpolate=True, **kwargs)
    line_45_top = line_45_bottom
    line_135_bottom = line_45_bottom
    line_135_top = line_45_bottom
    
    # Draw interactor line if applicable
    if trial:
        if trial[:-2] == "45_top":
            print("draw 45 top")
            # line_135_top.draw()
            return line_45_top
        elif trial[:-2] == "45_bottom":
            print("draw 45 bottom")
            # line_135_bottom.draw()
            return line_45_bottom
        elif trial[:-2] == "135_top":
            print("draw 135 top")
            # line_45_top.draw()
            return line_135_top
        elif trial[:-2] == "135_bottom":
            print("draw 135 bottom")
            # line_45_bottom.draw()
            return line_135_bottom
    
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
    
    
# Helper function to draw screen borders and other elements
def draw_screen_elements(trial, draw_occluder=False, exp_window=None, config=None,):
#                          left_border=None, right_border=None, top_border=None, bottom_border=None,
#                          line_45_top=None, line_45_bottom=None, line_135_top=None, line_135_bottom=None,
#                          occluder=None, fixation=None):
    
    fixation = create_cross_fixation(exp_window, **config["fixation"])
    occluder = create_occluder(exp_window, **config["occluder"])
    left_border, right_border, top_border, bottom_border = create_screen_borders(exp_window,
                                                                                         config["display"]["win_dims"],
                                                                                         config["display"]["square_size"])


    # line_map = {
    #     "45_top": create_interactor(win=exp_window, trial="45_top", ball_radius=config["ball"]["radius"], **config["interactor"]),
    #     "45_bottom":  create_interactor(win=exp_window, trial="45_bottom", ball_radius=config["ball"]["radius"], **config["interactor"]),
    #     "135_top":  create_interactor(win=exp_window, trial="135_top", ball_radius=config["ball"]["radius"], **config["interactor"]),
    #     "135_bottom":  create_interactor(win=exp_window, trial="135_bottom", ball_radius=config["ball"]["radius"], **config["interactor"]),
    #     }
    
    # Draw the fixation cross by drawing both lines
    def _draw_fixation():
        fixation[0].draw()
        fixation[1].draw()
    
    left_border.draw()
    right_border.draw()
    top_border.draw()
    bottom_border.draw()
    
    if trial:
        height = config["interactor"]["height"]
        width = config["interactor"]["width"]
        bounce_dist = get_bounce_dist(config["ball"]["radius"] + (width / 2 * 1.8)) # 1.8 factor is due to the that now we use an image, not sure if args are ok

        
        
    # Draw interactor line if applicable
    if trial:
        if trial[:-2] == "45_top":
            ImageStim(
                win=exp_window,
                # image="/Users/wiegerscheurer/Stimulus_material/135_flat_white.png", 
                image=config["interactor"]["path_45"],
                size=(height, height),
                pos= (-(bounce_dist), bounce_dist),
                opacity=1,
                interpolate=True).draw()
        elif trial[:-2] == "45_bottom":
            ImageStim(
                win=exp_window,
                # image="/Users/wiegerscheurer/Stimulus_material/135_flat_white.png", 
                image=config["interactor"]["path_45"],
                size=(height, height),
                pos=(bounce_dist, -(bounce_dist)),
                opacity=1,
                interpolate=True).draw()
        elif trial[:-2] == "135_top":
            ImageStim(
                win=exp_window,
                # image="/Users/wiegerscheurer/Stimulus_material/135_flat_white.png", 
                image=config["interactor"]["path_135"],
                size=(height, height),
                pos= ((bounce_dist), bounce_dist),
                opacity=1,
                interpolate=True).draw()
        elif trial[:-2] == "135_bottom":
            ImageStim(
                win=exp_window,
                # image="/Users/wiegerscheurer/Stimulus_material/135_flat_white.png", 
                image=config["interactor"]["path_135"],
                size=(height, height),
                pos=(-bounce_dist, -(bounce_dist)),
                opacity=1,
                interpolate=True).draw()
    
    # Draw occluder if needed
    if draw_occluder:
        occluder.draw()
        
    # fixation.draw()
    _draw_fixation()