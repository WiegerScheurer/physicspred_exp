#!/usr/bin/env python

import os.path as op
import sys
import time
import random
import numpy as np
import pandas as pd
import os
from psychopy import visual, event, core, gui
from local_exptools2.core import Session, Trial
from local_exptools2 import utils, stimuli, physics, analysis, plotting

class BehavTrial(Trial):
    """Trial class for physics prediction experiment with ball trajectory tasks."""
    
    def __init__(self, session, trial_nr, phase_durations, parameters=None, timing='seconds', 
                 trial_params=None, verbose=False, **kwargs):
        super().__init__(session, trial_nr, phase_durations, parameters, timing, **kwargs)
        
        self.verbose = verbose
        self.session = session
        self.trial_params = trial_params
        self.settings = session.settings
        
        self.phase_clock = session.clock
        
        # Extract trial parameters
        self.trial_type = trial_params['trial_type']
        self.trial_option = trial_params['trial_option']
        self.bounce = trial_params['bounce']
        self.ball_change = trial_params['ball_change']
        self.rand_bounce_direction = trial_params['phant_bounce_direction']
        self.ball_color_change = trial_params['ball_luminance']
        
        # Initialize trial-specific parameters
        self.ball_speed = session.ball_speeds[trial_nr]
        self.ball_start_color = session.ball_start_colors[trial_nr]
        self.edge_letter = self.trial_option[-1] if self.trial_option[:4] == "none" else self.trial_option.split("_")[2]
        self.edge = physics._flip_dir(next(option for option in session.edge_options if option.startswith(self.edge_letter)))
        
        # Ball movement tracking variables
        self.ballmov_time = 0
        self.enter_screen_time = None
        self.left_screen_time = None
        self.entered_screen = None
        self.bounce_moment = None
        self.correct_response = None
        self.crossed_fixation = False
        self.responded = False
        self.left_occluder = False
        self.ball_change_moment = None
        self.occluder_exit_moment = None
        self.pre_bounce_velocity = None
        self.bounced_phantomly = False
        
        # Set the starting position and velocity
        self.start_positions, self.directions, _, _, _, _ = utils.get_pos_and_dirs(
            session.settings['ball']['avg_speed'], 
            session.settings['display']['square_size'], 
            session.settings['ball']['spawn_spread'],
            session.settings['ball']['speed_change'], 
            session.settings['ball']['radius']
        )
        
        # Calculate changed ball color
        self.changed_ball_color = utils.oklab_to_rgb([
            (self.ball_start_color + self.ball_color_change), 0, 0
        ], psychopy_rgb=True)
        
        # # Create ball stimulus
        # self.ball = visual.Circle(
        #     win=session.win,
        #     radius=session.settings['ball']['radius'],
        #     fillColor=np.clip(utils.oklab_to_rgb([self.ball_start_color, 0, 0], psychopy_rgb=True)),
        #     lineWidth=0,
        #     pos=(0, 0),
        #     )
        self.ball = visual.Circle(win=session.win, 
                radius=session.settings['ball']['radius'],
                edges=64,
                fillColor="white",#config["ball_fillcolor"], 
                lineColor="white", #config["ball_linecolor"], 
                interpolate=True,
                opacity=1)

    def prepare_trial(self):
        """Prepare the trial before running."""
        # Store trial data
        self.session.exp_data["trial"].append(self.trial_nr + 1)
        self.session.exp_data["trial_type"].append(self.trial_type)
        self.session.exp_data["interactor"].append(self.trial_option)
        self.session.exp_data["bounce"].append(self.bounce)
        self.session.exp_data["ball_speed"].append(self.ball_speed)
        self.session.exp_data["ball_start_color"].append(self.ball_start_color)
        self.session.exp_data["ball_color_change"].append(float(self.ball_color_change))
        self.session.exp_data["target_color"].append(self.changed_ball_color if self.ball_change else None)
        
        # Initialize placeholders for data that will be filled during trial
        placeholders = [
            "bounce_moment", "target_onset", "abs_congruent", "sim_congruent",
            "response", "accuracy", "rt", "end_pos", "abs_rfup", "abs_rfright",
            "abs_rfdown", "abs_rfleft", "sim_rfup", "sim_rfright", 
            "sim_rfdown", "sim_rfleft"
        ]
        for key in placeholders:
            self.session.exp_data[key].append(None)
        
        # Add bounce direction data
        self.session.exp_data["random_bounce_direction"].append(
            self.rand_bounce_direction if self.bounce and self.trial_option[:4] == "none" else None
        )
        
        # Add remaining trial data
        self.session.exp_data["ball_change"].append(self.ball_change)
        self.session.exp_data["start_pos"].append(self.edge)
        
        # Set ball position
        self.ball.pos = np.array(self.start_positions[self.edge])
        self.velocity = np.array(self.directions[self.edge])
        
        if self.verbose:
            print(f"Trial {self.trial_nr + 1}: {self.trial_option}, Bounce: {self.bounce}, Edge: {self.edge}")

    def draw(self):
        """Draw the current phase of the trial."""
        # Draw the appropriate content based on the current phase
        if self.phase == 0:
            # Phase 0: Fixation Display
            self.draw_common_elements()
        elif self.phase == 1:
            # Phase 1: Interactor Display
            self.draw_common_elements()
            self.draw_interactor_elements()
        elif self.phase == 2:
            # Phase 2: Occluder Display
            self.draw_common_elements()
            self.draw_interactor_elements()
            self.session.occluder.draw()
        elif self.phase == 3:
            # Phase 3: Ball Movement
            self.handle_ball_movement()
            self.draw_common_elements()
            self.draw_interactor_elements()
            self.session.occluder.draw()
            self.ball.draw()

    def handle_ball_movement(self):
        """Handle the ball movement during phase 3."""
        frame_dur = self.session.settings['display']['frame_dur']
        decay_constant = self.session.settings['ball']['decay_constant']
        occluder_radius = self.session.settings['occluder']['radius']
        ball_radius = self.session.settings['ball']['radius']
        
        # Only update position if the trial is still running
        if not self.responded and not self.exit_phase:
            # Apply ball speed decay
            decay_factor = physics.calculate_decay_factor(
                self.ball_speed, self.ballmov_time, self.phase_durations[3], constant=decay_constant
            )
            self.velocity = [self.velocity[0] * decay_factor, self.velocity[1] * decay_factor]
            self.ball.pos += tuple([self.velocity[0], self.velocity[1]])
            
            # Update elapsed time
            self.ballmov_time += frame_dur
            
            # Track when ball enters and leaves screen
            square_size = self.session.settings['display']['square_size']
            if (np.linalg.norm(self.ball.pos) > square_size / 2) and self.phase_clock.getTime() < (self.phase_durations[3] / 2):
                self.enter_screen_time = self.phase_clock.getTime()
                self.entered_screen = True
            
            if (self.entered_screen and 
                self.enter_screen_time is not None and 
                self.left_screen_time is None and
                self.phase_clock.getTime() > (self.phase_durations[3] / 2) and
                (np.linalg.norm(self.ball.pos) > square_size / 2)):
                self.left_screen_time = self.phase_clock.getTime()
                if self.verbose:
                    screen_time = self.left_screen_time - self.enter_screen_time
                    print(f"LEFT SCREEN AT {self.left_screen_time:.3f}")
                    print(f"SCREEN TIME: {screen_time:.3f}")
            
            # Handle normal bounce
            if physics.will_cross_fixation(self.ball.pos, self.velocity, 1) and self.bounce and self.trial_option[:4] != "none":
                self.pre_bounce_velocity = np.max(np.abs(self.velocity)) if self.pre_bounce_velocity is None else self.pre_bounce_velocity
                if self.trial_option[:2] == "45":
                    if self.verbose:
                        print(f"BOUNCED on 45 at {self.phase_clock.getTime()}")
                    self.velocity = physics.collide(physics._flip_dir(self.edge), 45, self.pre_bounce_velocity)
                    self.bounce_moment = self.phase_clock.getTime()
                elif self.trial_option[:3] == "135":
                    if self.verbose:
                        print(f"BOUNCED on 135 at {self.phase_clock.getTime()}")
                    self.velocity = physics.collide(physics._flip_dir(self.edge), 135, self.pre_bounce_velocity)
                    self.bounce_moment = self.phase_clock.getTime()
                
                self.bounce = False
                self.crossed_fixation = True
            
            # Handle phantom bounce or fixation crossing
            if physics.will_cross_fixation(self.ball.pos, self.velocity, 1):
                if self.bounce and self.trial_option[:4] == "none":
                    self.pre_bounce_velocity = np.max(np.abs(self.velocity)) if self.pre_bounce_velocity is None else self.pre_bounce_velocity
                    
                    if self.verbose:
                        print("Phantom bounce")
                    
                    if self.rand_bounce_direction == "left":
                        if self.verbose:
                            print(f"BOUNCED LEFT at {self.phase_clock.getTime()}")
                        self.velocity = physics._dir_to_velocity(
                            physics._rotate_90(physics._flip_dir(self.edge), "left"), self.pre_bounce_velocity
                        )
                    elif self.rand_bounce_direction == "right":
                        if self.verbose:
                            print(f"BOUNCED RIGHT at {self.phase_clock.getTime()}")
                        self.velocity = physics._dir_to_velocity(
                            physics._rotate_90(physics._flip_dir(self.edge), "right"), self.pre_bounce_velocity
                        )
                    
                    self.bounced_phantomly = True
                    self.bounce_moment = self.phase_clock.getTime()
                
                elif not self.bounce and not self.bounced_phantomly:
                    self.bounce_moment = self.phase_clock.getTime()
                
                self.bounce = False
                self.crossed_fixation = True
                
                if self.verbose:
                    print(f"crossed fixation at {self.phase_clock.getTime()}")
            
            # Check if ball is leaving occluder
            if (np.linalg.norm(self.ball.pos) > (occluder_radius / 2) - (ball_radius * 2)
                and self.crossed_fixation
                and not self.left_occluder):
                
                if self.verbose:
                    print(f"occluder exit time: {self.phase_clock.getTime():.2f}")
                
                self.occluder_exit_moment = self.phase_clock.getTime()
                self.left_occluder = True
            
            # Handle ball changes after occluder exit
            if not self.bounce:
                if self.crossed_fixation and self.ball_change_moment is None and self.left_occluder:
                    if self.verbose:
                        print(f"ball_change_moment: {self.occluder_exit_moment}")
                    
                    self.ball_change_moment = self.occluder_exit_moment
                    self.session.exp_data["target_onset"][-1] = self.ball_change_moment if self.ball_change else None
                
                if self.session.task_choice == "Ball Hue" and self.crossed_fixation and self.left_occluder:
                    if self.ball_change:
                        self.ball.fillColor = self.changed_ball_color
            
            # Record ball direction and bounce moment
            ball_direction = physics.velocity_to_direction(self.velocity)
            self.session.exp_data["bounce_moment"][-1] = self.bounce_moment if physics._flip_dir(ball_direction) != self.edge else None
            self.session.exp_data["end_pos"][-1] = ball_direction

    def handle_responses(self):
        """Handle responses from the get_events method in the parent class."""
        if self.last_resp and not self.responded:
            if self.verbose:
                print(f"Response: {self.last_resp}")
            
            if not self.left_occluder:
                print(f"Wrong, too early")
                self.session.exp_data["response"][-1] = self.last_resp
                self.correct_response = None
                self.responded = True
            else:
                toets_moment = self.phase_clock.getTime()
                self.session.exp_data["response"][-1] = self.last_resp
                self.session.exp_data["rt"][-1] = toets_moment - self.ball_change_moment
                
                if self.ball_change and self.last_resp in ["x", "m"]:
                    if self.last_resp == self.session.button_order["brighter"]:
                        this_response = "brighter"
                    elif self.last_resp == self.session.button_order["darker"]:
                        this_response = "darker"
                    
                    if (this_response == "brighter" and self.ball_color_change > 0) or (this_response == "darker" and self.ball_color_change < 0):
                        print(f"Correct! detected a {this_response} ball in {round(toets_moment - self.ball_change_moment, 3)}s")
                        self.correct_response = True
                    elif self.ball_color_change == 0:
                        self.correct_response = None
                    elif (this_response == "brighter" and self.ball_color_change < 0) or (this_response == "darker" and self.ball_color_change > 0):
                        print(f"Wrong answer, the ball didn't become {this_response}")
                        self.correct_response = False
                    
                    self.session.exp_data["response"][-1] = this_response
                    self.session.exp_data["rt"][-1] = toets_moment - self.ball_change_moment
                    
                    if toets_moment < self.ball_change_moment:
                        print(f"Wrong, TOO EARLY")
                        self.correct_response = False
                else:
                    if self.ball_change:
                        print(f"Wrong, there was no change")
                        self.correct_response = False
                
                self.responded = True
                self.session.exp_data["accuracy"][-1] = self.correct_response

    def finish_trial(self):
        """Called at the end of the trial."""
        # Calculate predictions for the trial
        self.calculate_predictions()
        
        # Check if trial timed out without a response
        if self.phase_clock.getTime() > self.phase_durations[3] and not self.responded:
            if self.ball_change:
                self.correct_response = False
                if self.verbose:
                    print(f"Undetected ball change, there was a hue change of the ball")
            else:
                self.correct_response = None
            
            self.session.exp_data["accuracy"][-1] = self.correct_response

    def calculate_predictions(self):
        """Calculate predictions for ball path."""
        for hypothesis in ["abs", "sim"]:
            pred_to_input = physics.predict_ball_path(
                hypothesis=hypothesis,
                interactor=self.trial_option,
                start_pos=self.edge,
                end_pos=self.session.exp_data["end_pos"][-1],
                plot=False,
            )
            self.session.exp_data[f"{hypothesis}_congruent"][-1] = False
            for location in pred_to_input.keys():
                self.session.exp_data[f"{hypothesis}_rf{location}"][-1] = pred_to_input[location]
                if sum(pred_to_input[location]) == 2:
                    self.session.exp_data[f"{hypothesis}_congruent"][-1] = True

    def draw_common_elements(self):
        """Draw common screen elements."""
        for border in self.session.screen_borders:
            border.draw()
        
        for fix_line in self.session.cross_fix:
            fix_line.draw()
        
        if self.session.settings['display']['draw_grid']:
            for line in self.session.horizontal_lines + self.session.vertical_lines:
                line.draw()

    def draw_interactor_elements(self):
        """Draw interactor elements if applicable."""
        if self.trial_option[:4] != "none":
            interactor = stimuli.create_interactor(
                self.session.win, 
                self.trial_option,
                self.session.settings['ball']['radius'],
                width=self.session.settings['interactor']['width'],
                height=self.session.settings['interactor']['height'],
                path_45=self.session.settings['interactor']['path_45'],
                path_135=self.session.settings['interactor']['path_135'],
            )
            interactor.draw()


class BehavSession(Session):
    """Behaviour session for physics prediction experiment."""
    
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10):
        """Initialize BehavSession object."""
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)
        
        # Set up experiment parameters
        self.verbose = self.settings['experiment']['verbose']
        self.edge_options = ["up", "down", "left", "right"]
        self.task_choice = "Ball Hue"  # Default task
        self.give_feedback = True
        
        # Set up button order
        buttons = ["m", "x"]
        random.shuffle(buttons)
        self.button_order = {"brighter": buttons[0], "darker": buttons[1]}
        
        # Initialize experiment data dictionary
        self.exp_data = {par: [] for par in self.settings['experiment']['exp_parameters']}
        
        # Create design matrix
        self.design_matrix = utils.build_design_matrix(
            n_trials=n_trials,
            change_ratio=[True],
            ball_color_change_mean=self.settings['ball']['color_change_mean'],
            ball_color_change_sd=self.settings['ball']['color_change_sd'],
            verbose=self.verbose,
            neg_bias_factor=self.settings['ball']['neg_bias_factor']
        )
        
        # Generate ball speeds and starting colors
        self.ball_speeds = utils.bellshape_sample(
            float(self.settings['ball']['avg_speed']), 
            float(self.settings['ball']['natural_speed_variance']), 
            n_trials
        )
        self.ball_start_colors = utils.bellshape_sample(
            float(self.settings['ball']['start_color_mean']), 
            float(self.settings['ball']['start_color_sd']), 
            n_trials
        )
        
        # Generate inter-trial intervals
        self.itis = utils.truncated_exponential_decay(
            self.settings['timing']['min_iti'], 
            self.settings['timing']['max_iti'], 
            n_trials
        )
        
        # Set up experiment parameters
        self.start_experiment_time = None
        self.setup_experiment_elements()

    def setup_experiment_elements(self):
        """Set up visual elements for the experiment."""
        # Set up screen borders
        screen_size = self.settings['display']['square_size']
        border_width = self.settings['display']['border_width'] if 'border_width' in self.settings['display'] else 3
        
        self.screen_borders = []
        self.screen_borders.append(visual.Line(self.win, start=(-screen_size/2, -screen_size/2), end=(-screen_size/2, screen_size/2), lineWidth=border_width, lineColor="white"))  # Left
        self.screen_borders.append(visual.Line(self.win, start=(screen_size/2, -screen_size/2), end=(screen_size/2, screen_size/2), lineWidth=border_width, lineColor="white"))    # Right
        self.screen_borders.append(visual.Line(self.win, start=(-screen_size/2, -screen_size/2), end=(screen_size/2, -screen_size/2), lineWidth=border_width, lineColor="white"))  # Bottom
        self.screen_borders.append(visual.Line(self.win, start=(-screen_size/2, screen_size/2), end=(screen_size/2, screen_size/2), lineWidth=border_width, lineColor="white"))    # Top
        
        # Create diagonal lines
        self.screen_borders.append(visual.Line(self.win, start=(-screen_size/2, -screen_size/2), end=(screen_size/2, screen_size/2), lineWidth=border_width, lineColor="white"))   # 45 degrees
        self.screen_borders.append(visual.Line(self.win, start=(-screen_size/2, screen_size/2), end=(screen_size/2, -screen_size/2), lineWidth=border_width, lineColor="white"))   # 135 degrees
        
        # Create occluder
        self.occluder = visual.Circle(
            win=self.win,
            radius=self.settings['occluder']['radius'],
            fillColor="black",
            lineColor="black",
            pos=(0, 0)
        )
        
        # Create fixation cross
        cross_size = self.settings['display'].get('fixation_size', 10)
        line_width = self.settings['display'].get('fixation_width', 3)
        self.cross_fix = []
        self.cross_fix.append(visual.Line(self.win, start=(-cross_size/2, 0), end=(cross_size/2, 0), lineWidth=line_width, lineColor="red"))  # Horizontal
        self.cross_fix.append(visual.Line(self.win, start=(0, -cross_size/2), end=(0, cross_size/2), lineWidth=line_width, lineColor="red"))  # Vertical
        
        # Create grid lines if needed
        if self.settings['display']['draw_grid']:
            self.horizontal_lines = []
            self.vertical_lines = []
            grid_spacing = self.settings['display'].get('grid_spacing', 20)
            
            for i in range(-int(screen_size/2), int(screen_size/2) + grid_spacing, grid_spacing):
                if i != 0:  # Skip the center lines
                    self.horizontal_lines.append(visual.Line(
                        self.win, 
                        start=(-screen_size/2, i), 
                        end=(screen_size/2, i), 
                        lineWidth=1, 
                        lineColor="gray"
                    ))
                    self.vertical_lines.append(visual.Line(
                        self.win, 
                        start=(i, -screen_size/2), 
                        end=(i, screen_size/2), 
                        lineWidth=1, 
                        lineColor="gray"
                    ))

    def create_trials(self, durations=None, timing='seconds'):
        """Create trials for the experiment."""
        if durations is None:
            # Use default durations from settings
            durations = (
                self.settings['timing']['fixation_dur'],
                self.settings['timing']['interactor_dur'],
                self.settings['timing']['occluder_dur'],
                self.settings['timing']['ballmov_dur']
            )
        
        self.trials = []
        for trial_nr in range(self.n_trials):
            trial_params = {key: self.design_matrix.iloc[trial_nr][key] for key in self.design_matrix.columns}
            
            self.trials.append(
                BehavTrial(
                    session=self,
                    trial_nr=trial_nr,
                    phase_durations=durations,
                    parameters={'trial_nr': trial_nr},
                    timing=timing,
                    trial_params=trial_params,
                    verbose=self.verbose
                )
            )

    def run(self):
        """Run the experiment."""
        # Display welcome screen
        self.display_welcome_screen()
        
        # Display instructions
        self.display_instructions()
        
        # Start experiment
        self.start_experiment()
        self.start_experiment_time = time.time()
        
        # Run all trials
        for trial_idx, trial in enumerate(self.trials):
            trial.run()
            
            # Show break if needed
            feedback_freq = self.settings['experiment']['feedback_freq']
            if (trial_idx + 1) % feedback_freq == 0:
                self.show_feedback(trial_idx)
                
            # Wait for ITI
            self.wait_iti(trial_idx)
        
        # Save final data
        self.save_data()
        
        # Close the experiment
        self.close()

    def display_welcome_screen(self):
        """Display welcome screen."""
        welcome_text = visual.TextStim(
            self.win,
            text=f"Welcome! Today you will perform the Ball Hue task.\n\nPress 'Space' to continue.",
            color="white",
            pos=(0, 0),
            font="Arial",
            height=30,
        )
        
        instruction_read = ""
        while instruction_read != "SPACE":
            welcome_text.draw()
            self.win.flip()
            instruction_read = event.waitKeys(keyList=["space"])[0].upper()

    def display_instructions(self):
        """Display task instructions."""
        explanation_text = visual.TextStim(
            self.win,
            text=(
                "In this task, you will see a ball moving towards the\n"
                "center of the screen, where it passes behind a square.\n\n"
                "On top of this square you'll see a small red cross: +  \n"
                "Keep your eyes focused on this cross during the whole trial.\n\n"
                "The ball changes colour when behind the occluder \n\n"
                "You are challenged to detect these changes.\n\n"
                f"If the ball becomes brighter, press {self.button_order['brighter']}\n"
                f"If the ball becomes darker, press {self.button_order['darker']}\n"
                "Be as fast and accurate as possible!\n\n\n"
                f"We'll unveil your score every {self.settings['experiment']['feedback_freq']} trials.\n\n"
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
            explanation_text.draw()
            self.win.flip()
            ready_to_start = event.waitKeys(keyList=["space"])[0].upper()

    def wait_iti(self, trial_idx):
        """Wait for inter-trial interval."""
        if trial_idx < len(self.trials) - 1:  # Don't wait after last trial
            if self.verbose:
                print(f"Waiting for ITI: {self.itis[trial_idx]:.2f}s")
            core.wait(self.itis[trial_idx])

    def show_break(self, duration=10):
        """Show a break between trials."""
        longer_str = " longer" if duration > 10 else ""
        
        clock = core.Clock()
        countdown_text = visual.TextStim(self.win, text='', pos=(0, 0), height=20)
        break_text = visual.TextStim(
            self.win, 
            text=f'You deserve a{longer_str} break now.\n\nRemember: \n{self.button_order["brighter"]} for brighter\n{self.button_order["darker"]} for darker\n\nPress space to continue.', 
            pos=(0, 70), 
            height=30
        )
        
        while clock.getTime() < duration:
            remaining_time = duration - int(clock.getTime())
            countdown_text.text = f'\n\n\n\n\n\nBreak ends in {remaining_time} seconds'
            countdown_text.draw()
            break_text.draw()
            self.win.flip()
            
            keys = event.getKeys(keyList=['space'])
            if 'space' in keys:
                break

        # Clear any remaining key presses
        event.clearEvents()

    def show_feedback(self, trial_idx):
        """Show feedback to participant."""
        # Create DataFrame for intermediate analysis
        intermit_data = pd.DataFrame(self.exp_data)
        intermit_rt = np.mean(intermit_data["rt"].dropna())
        
        feedback_text = (
            f'Progress: {trial_idx + 1}/{self.n_trials}\n'
            f'Detected changes: {(analysis.get_hit_rate(intermit_data, sim_con=None, expol_con=None)*100):.2f}%\n'
            f'Average speed: {intermit_rt:.2f}s\n\n'
            f'Remember: \n{self.button_order["brighter"]} for brighter\n{self.button_order["darker"]} for darker'
        )
        
        # Save intermediate data
        subject = self.output_str
        datadir = self.settings['paths']['datadir']
        os.makedirs(f"{datadir}/{subject}", exist_ok=True)
        intermit_data.to_csv(f"{datadir}/{subject}/intermit_data.csv", float_format="%.8f")
        
        # Display feedback
        feedback = visual.TextStim(
            self.win, text=feedback_text, color="white", pos=(0, 150), height=30
        )
        
        self.draw_common_elements()
        feedback.draw()
        self.win.flip()
        core.wait(self.settings['timing']['feedback_dur'])

    def start_experiment(self):
        """Initialize experiment timing."""
        self.experiment_clock = core.Clock()
        self.experiment_start_time = time.time()

    def save_data(self):
        """Save experiment data."""
        # Create final dataframe
        df = pd.DataFrame(self.exp_data)
        
        # Save performance data
        subject_id = self.output_str
        task_name = self.task_choice.lower().replace(" ", "_")
        utils.save_performance_data(subject_id, task_name, df, base_dir=self.settings['paths']['datadir'])
        utils.save_performance_data(subject_id, task_name, self.design_matrix, design_matrix=True, base_dir=self.settings['paths']['datadir'])
        
        # Record timing information
        end_time = time.time()
        elapsed_time = end_time - self.experiment_start_time
        timing_df = pd.DataFrame({"n_trials": [self.n_trials], "time_elapsed": [elapsed_time]})
        timing_df.to_csv(f"{self.settings['paths']['datadir']}/{subject_id}/timing.csv")

    def close(self):
        """Close the experiment window."""
        self.win.close()
        core.quit()


if __name__ == "__main__":
    # Set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Run Physics Prediction Experiment')
    parser.add_argument('--subject', type=str, default=None, help='Subject ID')
    parser.add_argument('--n_trials', type=int, default=None, help='Number of trials')
    parser.add_argument('--settings', type=str, default='behav_settings.yml', help='Settings file')
    args = parser.parse_args()
    
    # Set subject ID
    if args.subject is None:
        subject_id = f"sub-{random.randint(0, 999999):06.0f}"
    else:
        subject_id = args.subject
    
    # Set number of trials
    n_trials = args.n_trials if args.n_trials is not None else 10
    
    # Get settings file path
    settings_file = os.path.join(os.path.dirname(__file__), os.pardir, args.settings)
    
    # Create and run session
    session = BehavSession(
        output_str=subject_id,
        output_dir=None,
        settings_file=settings_file,
        n_trials=n_trials
    )
    
    # Create trials with proper durations
    session.create_trials(durations=(
        session.settings['timing']['fixation_dur'],
        session.settings['timing']['interactor_dur'],
        session.settings['timing']['occluder_dur'],
        session.settings['timing']['ballmov_dur']
    ))
    
    # Run experiment
    session.run()