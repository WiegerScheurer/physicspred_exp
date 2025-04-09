# Import necessary libraries
from psychopy import visual, core, event, gui, data
import numpy as np
import random
import pandas as pd
import os
import sys
import time
from omegaconf import OmegaConf

# from functions.utilities import (
from utils import (
    setup_folders,
    save_performance_data,
    # interpolate_color,
    build_design_matrix,
    bellshape_sample,
    ordinal_sample,
    oklab_to_rgb,
    truncated_exponential_decay,
    get_pos_and_dirs,
    check_balance,
)
# from functions.physics import (
from physics import (
    check_collision,
    collide,
    velocity_to_direction,
    predict_ball_path,
    _flip_dir,
    _rotate_90,
    _dir_to_velocity,
    will_cross_fixation,
    calculate_decay_factor
)


from analysis import get_hit_rate

import stimuli

# --- TRIAL COMPONENT FUNCTIONS ---

def show_break(win, duration=10, button_order={"brighter": "m", "darker": "x"}):
    """Display a break screen with countdown timer and button reminders."""
    longer_str = " longer" if duration > 10 else ""
    
    clock = core.Clock()
    countdown_text = visual.TextStim(win, text='', pos=(0, 0), height=20)
    break_text = visual.TextStim(
        win, 
        text=f'You deserve a{longer_str} break now.\n\nRemember: \n{button_order["brighter"]} for brighter\n{button_order["darker"]} for darker\n\n'
             f'Press space to continue.', 
        pos=(0, 70), 
        height=30
    )
    
    while clock.getTime() < duration:
        remaining_time = duration - int(clock.getTime())
        countdown_text.text = f'\n\n\n\n\n\nBreak ends in {remaining_time} seconds'
        countdown_text.draw()
        break_text.draw()
        win.flip()
        
        keys = event.getKeys(keyList=['space'])
        if 'space' in keys:
            break

    # Clear any remaining key presses
    event.clearEvents()


def setup_experiment(exp_window, config_filename:str="config_lumin.yaml"):
    """Set up experiment parameters, load configuration, and return essential objects."""
    # Load configuration file
    config_path = os.path.join(os.path.dirname(__file__), os.pardir, config_filename)
    config = OmegaConf.load(config_path)
    
    # Set up experiment directory
    _thisDir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_thisDir)
    
    # Set up experiment info
    expName = "IPE_SpatTemp_Pred_Behav"
    expInfo = {
        "participant": f"sub-{random.randint(0, 999999):06.0f}",
        "session": "001",
        "task": ["Ball Hue", "Ball Hiccup", "Ball Speed Change", "Fixation Hue Change"],
        "feedback": ["No", "Yes"],
    }
    
    # Show participant dialog and get input
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # User pressed cancel
    
    # Add timestamp and experiment info
    expInfo["date"] = data.getDateStr()
    expInfo["expName"] = expName
    expInfo["psychopyVersion"] = config.experiment.psychopy_version
    give_feedback = True  # if expInfo["feedback"] == "Yes" else False
    
    # Set up button mappings
    buttons = ["m", "x"]
    random.shuffle(buttons)
    button_order = {"brighter": buttons[0], "darker": buttons[1]}
    
    fixation = stimuli.create_cross_fixation(exp_window, **config["fixation"])
    occluder = stimuli.create_occluder(exp_window, **config["occluder"])
    left_border, right_border, top_border, bottom_border = stimuli.create_screen_borders(exp_window,
                                                                                         config["display"]["win_dims"],
                                                                                         config["display"]["square_size"])
    
    # Import task components
    # from objects.task_components import (
    from stimuli import (
        # win,
        # ball,
        # left_border,
        # right_border,
        # top_border,
        # bottom_border,
        # line_45_top,
        # line_45_bottom,
        # line_135_top,
        # line_135_bottom,
        # occluder,
        # fixation,
        # horizontal_lines,
        # vertical_lines,    
        draw_screen_elements,
    )
    
    # # Create a mapping for lines to simplify code later
    # line_map = {
    #     "45_top": line_45_top,
    #     "45_bottom": line_45_bottom,
    #     "135_top": line_135_top,
    #     "135_bottom": line_135_bottom,
    # }
    
    line_map = {
        "45_top": stimuli.create_interactor(exp_window, "45_top", config["ball"]["radius"], **config["interactor"]),
        "45_bottom":  stimuli.create_interactor(exp_window, "45_bottom", config["ball"]["radius"], **config["interactor"]),
        "135_top":  stimuli.create_interactor(exp_window, "135_top", config["ball"]["radius"], **config["interactor"]),
        "135_bottom":  stimuli.create_interactor(exp_window, "135_bottom", config["ball"]["radius"], **config["interactor"]),
    }
    
    # Hide mouse cursor and set up window
    exp_window.mouseVisible = False
    
    # Get and store frame rate of monitor
    expInfo["frameRate"] = exp_window.getActualFrameRate()
    frameDur = 1.0 / round(expInfo["frameRate"]) if expInfo["frameRate"] != None else 1.0 / 120.0
    refreshRate = round(expInfo["frameRate"], 0) if expInfo["frameRate"] != None else None
    
    # Build a ball
    ball = stimuli.create_ball(win=exp_window, ball_radius=config.ball.radius) #, color=config.ball.start_color_mean)
    # Create components dictionary to return
    components = {
        "config": config,
        "expInfo": expInfo,
        # "win": win,
        "win": exp_window,
        "ball": ball,
        "button_order": button_order,
        "line_map": line_map,
        "frameDur": frameDur,
        "refreshRate": refreshRate,
        "draw_screen_elements": draw_screen_elements,
        "give_feedback": give_feedback
    }
    
    return components

def create_design(components, verbose, n_trials):
    """Create experiment design matrix and extract trial parameters."""
    config = components["config"]
    verbose = config.experiment.verbose
    n_trials = config.experiment.n_trials
    
    # Create experiment design matrix
    design_matrix = build_design_matrix(
        n_trials=n_trials,
        change_ratio=[True],
        ball_color_change_mean=config.ball.color_change_mean,
        ball_color_change_sd=config.ball.color_change_sd,
        verbose=verbose,
        neg_bias_factor=config.ball.neg_bias_factor,
    )
    
    if verbose:
        check_balance(design_matrix)
        
    print(f'Ball changes: {list(ordinal_sample(config.ball.color_change_mean, config.ball.color_change_sd, n_elements=5, round_decimals=3, neg_bias_factor=config.ball.neg_bias_factor))}')
    
    # Extract trial parameters from design matrix
    trial_types = list(design_matrix["trial_type"])
    trials = list(design_matrix["trial_option"])
    bounces = list(design_matrix["bounce"])
    rand_bounce_directions = list(design_matrix["phant_bounce_direction"])
    ball_changes = list(design_matrix["ball_change"])
    ball_color_changes = list(design_matrix["ball_luminance"])
    
    # Generate ball speeds and starting colors
    ball_speeds = bellshape_sample(float(config.ball.avg_speed), float(config.ball.natural_speed_variance), n_trials)
    ball_start_colors = bellshape_sample(float(config.ball.start_color_mean), float(config.ball.start_color_sd), n_trials)
    
    # Generate inter-trial intervals
    itis = truncated_exponential_decay(config.timing.min_iti, config.timing.max_iti, n_trials)
    
    # Create trial parameters dictionary
    trial_params = {
        "design_matrix": design_matrix,
        "trial_types": trial_types,
        "trials": trials,
        "bounces": bounces,
        "rand_bounce_directions": rand_bounce_directions,
        "ball_changes": ball_changes,
        "ball_color_changes": ball_color_changes,
        "ball_speeds": ball_speeds,
        "ball_start_colors": ball_start_colors,
        "itis": itis
    }
    
    return trial_params


def initialize_data_structure(components, trial_params):
    """Initialize the data structure to store experiment results."""
    config = components["config"]
    
    # Initialize experiment data dictionary
    exp_parameters = config.experiment.exp_parameters
    exp_data = {par: [] for par in exp_parameters}
    
    return exp_data

def show_instructions(components):
    """Display welcome and instruction screens."""
    win = components["win"]
    button_order = components["button_order"]
    config = components["config"]
    
    # Display welcome screen
    welcome_text = visual.TextStim(
        win,
        text=f"Welcome! Today you will perform the Ball Hue task.\n\nPress 'Space' to continue.",
        color="white",
        pos=(0, 0),
        font="Arial",
        height=30,
    )
    
    instruction_read = ""
    while instruction_read != "SPACE":
        welcome_text.draw()
        win.flip()
        instruction_read = event.waitKeys(keyList=["space"])[0].upper()
    
    # Display task instructions
    explanation_text_speed = visual.TextStim(
        win,
        text=(
            "In this task, you will see a ball moving towards the\n"
            "center of the screen, where it passes behind a square.\n\n"
            "On top of this square you'll see a small red cross: +  \n"
            "Keep your eyes focused on this cross during the whole trial.\n\n"
            "The ball changes colour when behind the occluder \n\n"
            "You are challenged to detect these changes.\n\n"
            f"If the ball becomes brighter, press {button_order['brighter']}\n"
            f"If the ball becomes darker, press {button_order['darker']}\n"
            "Be as fast and accurate as possible!\n\n\n"
            f"We'll unveil your score every {config.experiment.feedback_freq} trials.\n\n"
            "Press 'Space' to start."
        ),
        color="white",
        font="Arial",
        pos=(0, 0),
        height=30,
        wrapWidth=1000,
    )
    
    ready_to_start = ""
    while ready_to_start != "SPACE":
        explanation_text_speed.draw()
        win.flip()
        ready_to_start = event.waitKeys(keyList=["space"])[0].upper()

def setup_trial(components, trial_params, trial_number):
    """Set up parameters for a specific trial."""
    config = components["config"]
    verbose = config.experiment.verbose
    
    # Get trial parameters
    trial = trial_params["trials"][trial_number]
    bounce = trial_params["bounces"][trial_number]
    rand_bounce_direction = trial_params["rand_bounce_directions"][trial_number]
    ball_change = trial_params["ball_changes"][trial_number]
    ball_color_change = trial_params["ball_color_changes"][trial_number]
    this_ball_speed = trial_params["ball_speeds"][trial_number]
    ball_start_color = trial_params["ball_start_colors"][trial_number]
    this_iti = trial_params["itis"][trial_number]
    
    # Get positions and directions for ball movement
    start_positions, directions, fast_directions, slow_directions, skip_directions, wait_directions = get_pos_and_dirs(
        config.ball.avg_speed, config.display.square_size, config.ball.spawn_spread, 
        config.ball.speed_change, config.ball.radius
    )
    
    # Calculate changed ball color
    changed_ball_color = oklab_to_rgb([(ball_start_color + ball_color_change), 0, 0], psychopy_rgb=True)
    
    # Extract start position letter from trial name
    if trial[:4] == "none":
        edge_letter = trial[-1]
    else:
        edge_letter = trial.split("_")[2]
        
    # Find the full edge option string
    edge_options = ["up", "down", "left", "right"]
    edge = _flip_dir(next(option for option in edge_options if option.startswith(edge_letter)))
    
    # Initialize trial variables
    trial_data = {
        "trial": trial,
        "bounce": bounce,
        "rand_bounce_direction": rand_bounce_direction,
        "ball_change": ball_change,
        "ball_color_change": ball_color_change,
        "this_ball_speed": this_ball_speed,
        "ball_start_color": ball_start_color,
        "changed_ball_color": changed_ball_color,
        "this_iti": this_iti,
        "edge": edge,
        "start_positions": start_positions,
        "directions": directions,
        "ball_change_delay": 0,
        "bounce_moment": None,
        "correct_response": None,
        "crossed_fixation": False,
        "responded": False,
        "left_occluder": False,
        "ball_change_moment": None,
        "occluder_exit_moment": None,
        "hue_changed": False,
        "hue_changed_back": False,
        "pre_bounce_velocity": None,
        "bounced_phantomly": False,
        "enter_screen_time": None,
        "left_screen_time": None,
        "entered_screen": None
    }
    
    if verbose:
        print(f"Trial: {trial}")
        print(f"Edge letter: {edge_letter}")
        print(f"Actual edge: {edge}")
        print(f"Ball will bounce: {bounce}")
        print(f"Target trial: {ball_change}")
    
    return trial_data

def run_fixation_phase(components, trial_data, trial_clock, win, config):
    """Run the fixation cross display phase."""
    win = components["win"]
    # draw_screen_elements = components["draw_screen_elements"]
    config = components["config"]
    verbose = config.experiment.verbose
    
    # FIXATION CROSS DISPLAY
    stimuli.draw_screen_elements(trial=None, draw_occluder=False, exp_window=win, config=config)
    
    refreshInformation = visual.TextStim(
        win=win,
        name="refreshInformation",
        text="",
        font="Arial",
        pos=(0, 0),
        height=0.05,
        wrapWidth=None,
        ori=0.0,
        color="white",
        colorSpace="rgb",
        opacity=None,
        languageStyle="LTR",
        depth=-1.0,
    )
    refreshInformation.setAutoDraw(True)
    win.flip()
    
    if verbose:
        print(f"Exact Fixation time: {trial_clock.getTime()}s")
    core.wait(config.timing.fixation_dur)


def run_interactor_phase(components, trial_data, trial_clock, win, config):
    """Run the interactor line display phase."""
    win = components["win"]
    # draw_screen_elements = components["draw_screen_elements"]
    config = components["config"]
    verbose = config.experiment.verbose
    
    # INTERACTOR LINE DISPLAY
    # draw_screen_elements(trial_data["trial"], draw_grid=config.display.draw_grid)
    print(f"Drawing interactor line: {trial_data['trial']}")
    stimuli.draw_screen_elements(trial=trial_data["trial"], draw_occluder=False, exp_window=win, config=config)

    win.flip()
    
    if verbose:
        print(f"Exact interactor time: {trial_clock.getTime()}s")
    core.wait(config.timing.interactor_dur)


def run_occluder_phase(components, trial_data, trial_clock, win, config):
    """Run the occluder display phase."""
    win = components["win"]
    # draw_screen_elements = components["draw_screen_elements"]
    config = components["config"]
    verbose = config.experiment.verbose
    
    # OCCLUDER DISPLAY
    # draw_screen_elements(trial_data["trial"], draw_occluder=True, draw_grid=config.display.draw_grid)
    stimuli.draw_screen_elements(trial=trial_data["trial"], draw_occluder=True, exp_window=win, config=config)
    win.flip()
    
    if verbose:
        print(f"Exact occluder time: {trial_clock.getTime()}s")
    core.wait(config.timing.occluder_dur)

def setup_ball_initial_state(components, trial_data):
    """Set the initial position and velocity of the ball."""
    ball = components["ball"]
    
    # Set ball position and velocity
    ball.pos = np.array(trial_data["start_positions"][trial_data["edge"]])
    velocity = np.array(trial_data["directions"][trial_data["edge"]])
    
    # Set initial ball color
    ball_start_color = trial_data["ball_start_color"]
    ball.color = np.clip(oklab_to_rgb([ball_start_color, 0, 0], psychopy_rgb=True), -1, 1)
    
    return velocity


def calculate_trial_duration(components):
    """Calculate the total duration of a trial."""
    config = components["config"]
    
    # Calculate trial duration
    trial_duration = (
        config.timing.fixation_dur + 
        config.timing.interactor_dur + 
        config.timing.occluder_dur + 
        config.timing.ballmov_dur
    )
    
    return trial_duration


def store_trial_data(exp_data, components, trial_params, trial_data, trial_number):
    """Store the initial trial data in the experiment data structure."""
    # Store trial characteristics
    trial_no = len(exp_data["trial"]) + 1
    exp_data["trial"].append(trial_no)
    exp_data["trial_type"].append(trial_params["trial_types"][trial_number])
    exp_data["interactor"].append(trial_data["trial"])
    exp_data["bounce"].append(trial_data["bounce"])
    exp_data["ball_speed"].append(trial_data["this_ball_speed"])
    exp_data["ball_start_color"].append(trial_data["ball_start_color"])
    exp_data["ball_color_change"].append(float(trial_data["ball_color_change"]))
    exp_data["target_color"].append(trial_data["changed_ball_color"] if trial_data["ball_change"] else None)
    
    # Initialize placeholders for data that will be filled during trial
    placeholders = [
        "bounce_moment", "target_onset", "abs_congruent", "sim_congruent",
        "response", "accuracy", "rt", "end_pos", "abs_rfup", "abs_rfright",
        "abs_rfdown", "abs_rfleft", "sim_rfup", "sim_rfright", 
        "sim_rfdown", "sim_rfleft"
    ]
    for key in placeholders:
        exp_data[key].append(None)
    
    # Add bounce direction data
    exp_data["random_bounce_direction"].append(
        trial_data["rand_bounce_direction"] if trial_data["bounce"] and trial_data["trial"][:4] == "none" else None
    )
    
    # Add remaining trial data
    exp_data["ball_change"].append(trial_data["ball_change"])
    exp_data["start_pos"].append(trial_data["edge"])
    
    return exp_data

def handle_normal_bounce(trial_data, velocity, trial_clock):
    """Handle normal bounce physics when ball crosses the interactor line."""
    pre_bounce_velocity = trial_data["pre_bounce_velocity"]
    trial = trial_data["trial"]
    edge = trial_data["edge"]
    
    pre_bounce_velocity = np.max(np.abs(velocity)) if pre_bounce_velocity is None else pre_bounce_velocity
    
    if trial[:2] == "45":
        print(f"BOUNCED on 45 at {trial_clock.getTime()}")
        velocity = collide(_flip_dir(edge), 45, pre_bounce_velocity)
        bounce_moment = trial_clock.getTime()
    elif trial[:3] == "135":
        print(f"BOUNCED on 135 at {trial_clock.getTime()}")
        velocity = collide(_flip_dir(edge), 135, pre_bounce_velocity)
        bounce_moment = trial_clock.getTime()
    
    return velocity, bounce_moment, pre_bounce_velocity


def handle_phantom_bounce(trial_data, velocity, trial_clock):
    """Handle phantom bounce physics for trials with no interactor."""
    pre_bounce_velocity = trial_data["pre_bounce_velocity"]
    rand_bounce_direction = trial_data["rand_bounce_direction"]
    edge = trial_data["edge"]
    
    pre_bounce_velocity = np.max(np.abs(velocity)) if pre_bounce_velocity is None else pre_bounce_velocity
    
    if rand_bounce_direction == "left":
        print(f"BOUNCED LEFT at {trial_clock.getTime()}")
        velocity = _dir_to_velocity(
            _rotate_90(_flip_dir(edge), "left"), pre_bounce_velocity
        )
    elif rand_bounce_direction == "right":
        print(f"BOUNCED RIGHT at {trial_clock.getTime()}")
        velocity = _dir_to_velocity(
            _rotate_90(_flip_dir(edge), "right"), pre_bounce_velocity
        )
    
    bounce_moment = trial_clock.getTime()
    
    return velocity, bounce_moment, pre_bounce_velocity, True


def process_ball_movement(components, trial_data, velocity, trial_clock, trial_duration, exp_data, this_ballmov_time=None):
    """Process ball movement for a single frame."""
    config = components["config"]
    ball = components["ball"]
    win = components["win"]
    # draw_screen_elements = components["draw_screen_elements"]
    expInfo = components["expInfo"]
    button_order = components["button_order"]
    
    frame_dur = config.display.frame_dur
    square_size = config.display.square_size
    occluder_radius = config.occluder.radius
    ball_radius = config.ball.radius
    
    # Apply ball speed decay
    decay_factor = calculate_decay_factor(
        trial_data["this_ball_speed"], this_ballmov_time, config.timing.ballmov_dur, constant=config.ball.decay_constant
    )
    
    this_ballmov_time += frame_dur # maybe before? decay fact comp?


    velocity = [velocity[0] * decay_factor, velocity[1] * decay_factor]
    ball.pos += tuple([velocity[0] * 1, velocity[1] * 1])  # Using skip_factor=1
    
    # Track when ball enters and leaves screen
    if (np.linalg.norm(ball.pos) > square_size / 2) and trial_clock.getTime() < (trial_duration // 2):
        trial_data["enter_screen_time"] = trial_clock.getTime()
        trial_data["entered_screen"] = True
    
    if (trial_data["entered_screen"] and 
        trial_data["enter_screen_time"] is not None and 
        trial_data["left_screen_time"] is None and
        trial_clock.getTime() > (trial_duration // 2) and
        (np.linalg.norm(ball.pos) > square_size // 2)):
        trial_data["left_screen_time"] = trial_clock.getTime()
        if config.experiment.verbose:
            screen_time = trial_data["left_screen_time"] - trial_data["enter_screen_time"]
            print(f"LEFT SCREEN AT {trial_data['left_screen_time']:.3f}")
            print(f"SCREEN TIME: {screen_time:.3f}")
    
    # Handle normal bounce
    if will_cross_fixation(ball.pos, velocity, 1) and trial_data["bounce"] and trial_data["trial"][:4] != "none":
        velocity, trial_data["bounce_moment"], trial_data["pre_bounce_velocity"] = handle_normal_bounce(
            trial_data, velocity, trial_clock
        )
        trial_data["bounce"] = False
        trial_data["crossed_fixation"] = True
    
    # Draw the current frame
    ball.draw()
    # draw_screen_elements(trial_data["trial"], draw_occluder=True, draw_grid=config.display.draw_grid)
    stimuli.draw_screen_elements(trial=trial_data["trial"], draw_occluder=True, exp_window=win, config=config)
    win.flip()
    core.wait(frame_dur)  # This needs to be here to make the ball move at the right speed
    
    # Handle phantom bounce or fixation crossing
    if will_cross_fixation(ball.pos, velocity, 1):
        if trial_data["bounce"] and trial_data["trial"][:4] == "none":
            velocity, trial_data["bounce_moment"], trial_data["pre_bounce_velocity"], trial_data["bounced_phantomly"] = handle_phantom_bounce(
                trial_data, velocity, trial_clock
            )
        
        elif not trial_data["bounce"] and not trial_data["bounced_phantomly"]:
            trial_data["bounce_moment"] = trial_clock.getTime()
        
        trial_data["bounce"] = False
        trial_data["crossed_fixation"] = True
        
        if config.experiment.verbose:
            print(f"crossed fixation at {trial_clock.getTime()}")
    
    # Check if ball is leaving occluder
    if (np.linalg.norm(ball.pos) > (occluder_radius / 2) - (ball_radius * 2)
        and trial_data["crossed_fixation"]
        and not trial_data["left_occluder"]):
        
        if config.experiment.verbose:
            print(f"occluder exit time: {trial_clock.getTime():.2f}")
        print(f"LEAVING OCCLUDER NOW!! exit time: {trial_clock.getTime():.2f}")
        
        trial_data["occluder_exit_moment"] = trial_clock.getTime()
        trial_data["left_occluder"] = True
    
    # Handle ball changes after occluder exit
    if trial_data["bounce"] == False:
        if trial_data["crossed_fixation"] and trial_data["ball_change_moment"] is None and trial_data["left_occluder"]:
            
            if config.experiment.verbose:
                print(f"ball_change_moment: {trial_data['occluder_exit_moment'] + trial_data['ball_change_delay']}")
            
            trial_data["ball_change_moment"] = trial_data["occluder_exit_moment"] + trial_data["ball_change_delay"]
            exp_data["target_onset"][-1] = trial_data["ball_change_moment"] if trial_data["ball_change"] else None
        
        if expInfo["task"] == "Ball Hue" and trial_data["crossed_fixation"] and trial_data["ball_change_moment"] is None:
            ball.color = trial_data["changed_ball_color"]
    
    # Record ball direction and bounce moment
    ball_direction = velocity_to_direction(velocity)
    exp_data["bounce_moment"][-1] = trial_data["bounce_moment"] if _flip_dir(ball_direction) != trial_data["edge"] else None
    exp_data["end_pos"][-1] = ball_direction
    
    return velocity, ball_direction, exp_data, trial_data, this_ballmov_time


def handle_responses(components, trial_data, trial_clock, exp_data):
    """Handle user responses during the trial."""
    config = components["config"]
    expInfo = components["expInfo"]
    button_order = components["button_order"]
    
    # Check for key presses
    keys = event.getKeys(["space", "x", "m", "escape"])
    
    if "escape" in keys:
        print("ESCAPE PRESSED")
        components["win"].close()
        core.quit()
    
    if keys and not trial_data["responded"]:
        if config.experiment.verbose:
            print(f"Response: {keys[0]}")
        
        if not trial_data["left_occluder"]:
            print(f"Wrong, too early")
            exp_data["response"][-1] = keys[0]
            trial_data["correct_response"] = None
            trial_data["responded"] = True
        else:
            toets_moment = trial_clock.getTime()
            exp_data["response"][-1] = keys[0]
            exp_data["rt"][-1] = toets_moment - trial_data["ball_change_moment"]
            
            if trial_data["ball_change"] and keys[0] in ["x", "m"]:
                ball_change_type = "H"  # Assuming Ball Hue is default
                
                if keys[0] == button_order["brighter"]:
                    this_response = "brighter"
                elif keys[0] == button_order["darker"]:
                    this_response = "darker"
                
                if (this_response == "brighter" and trial_data["ball_color_change"] > 0) or (this_response == "darker" and trial_data["ball_color_change"] < 0):
                    print(f"Correct! detected a {this_response} ball in {round(toets_moment - trial_data['ball_change_moment'], 3)}s")
                    trial_data["correct_response"] = True
                elif trial_data["ball_color_change"] == 0:
                    trial_data["correct_response"] = None
                elif (this_response == "brighter" and trial_data["ball_color_change"] < 0) or (this_response == "darker" and trial_data["ball_color_change"] > 0):
                    print(f"Wrong answer, the ball didn't become {this_response}")
                    trial_data["correct_response"] = False
                
                exp_data["response"][-1] = this_response
                exp_data["rt"][-1] = toets_moment - trial_data["ball_change_moment"]
                
                if toets_moment < trial_data["ball_change_moment"]:
                    print(f"Wrong, TOO EARLY")
                    trial_data["correct_response"] = False
            else:
                if trial_data["ball_change"]:
                    print(f"Wrong, there was no change")
                    trial_data["correct_response"] = False
            
            trial_data["responded"] = True
            
    elif trial_clock.getTime() > calculate_trial_duration(components) and not trial_data["responded"]:
        if trial_data["ball_change"]:
            ball_direction = exp_data["end_pos"][-1]
            feedback_text = f"Undetected ball change, there was a hue change of the {ball_direction}ward ball" if components["give_feedback"] else ""
            trial_data["correct_response"] = False
            if config.experiment.verbose:
                print(feedback_text)
        else:
            feedback_text = ""
            trial_data["correct_response"] = None
            if config.experiment.verbose:
                print(feedback_text)
    
    exp_data["accuracy"][-1] = trial_data["correct_response"]
    
    return exp_data, trial_data


def calculate_predictions(trial_data, exp_data):
    """Calculate predictions for ball path."""
    for hypothesis in ["abs", "sim"]:
        pred_to_input = predict_ball_path(
            hypothesis=hypothesis,
            interactor=trial_data["trial"],
            start_pos=trial_data["edge"],
            end_pos=exp_data["end_pos"][-1],
            plot=False,
        )
        exp_data[f"{hypothesis}_congruent"][-1] = False
        for location in pred_to_input.keys():
            exp_data[f"{hypothesis}_rf{location}"][-1] = pred_to_input[location]
            if sum(pred_to_input[location]) == 2:
                exp_data[f"{hypothesis}_congruent"][-1] = True
                
    return exp_data

def show_feedback(components, trial_number, exp_data):
    """Display feedback to participant at specified intervals."""
    config = components["config"]
    win = components["win"]
    button_order = components["button_order"]
    datadir = config.paths.datadir
    expInfo = components["expInfo"]
    feedback_freq = config.experiment.feedback_freq
    n_trials = config.experiment.n_trials
    feedback_dur = config.timing.feedback_dur
    
    if (trial_number + 1) % feedback_freq == 0:
        intermit_data = pd.DataFrame(exp_data)
        intermit_rt = np.mean(intermit_data["rt"].dropna())
        feedback_text = (
            f'Progress: {trial_number + 1}/{n_trials}\n'
            f'Detected changes: {(get_hit_rate(intermit_data, sim_con=None, expol_con=None)*100):.2f}%\n'
            f'Average speed: {intermit_rt:.2f}s\n\n'
            f'Remember: \n{button_order["brighter"]} for brighter\n{button_order["darker"]} for darker'
        )
        
        subject = expInfo["participant"]
        os.makedirs(f"{datadir}/{subject}", exist_ok=True)
        intermit_data.to_csv(f"{datadir}/{subject}/intermit_data.csv", float_format="%.8f")
        
        if (trial_number + 1) % (n_trials // 2) == 0 and (trial_number + 1 != n_trials):
            # Halfway break
            show_break(win, duration=30, button_order=button_order)
            
            feedback = visual.TextStim(
                win, text=feedback_text, color="white", pos=(0, 150), height=30
            )
            # components["draw_screen_elements"](None)
            stimuli.draw_screen_elements(trial=None, draw_occluder=False, exp_window=win, config=config)
            feedback.draw()
            win.flip()
            core.wait(feedback_dur)
        else:
            # Regular break
            show_break(win, duration=10, button_order=button_order)
            
            feedback = visual.TextStim(
                win, text=feedback_text, color="white", pos=(0, 150), height=30
            )
            # components["draw_screen_elements"](None)
            stimuli.draw_screen_elements(trial=None, draw_occluder=False, exp_window=win, config=config)
            feedback.draw()
            win.flip()
            core.wait(feedback_dur)