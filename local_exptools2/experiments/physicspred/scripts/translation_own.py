#!/usr/bin/env python

import os.path as op
import sys
# sys.path.insert(0, '/Users/wiegerscheurer/repos/exptools2/local_exptools2/')
# print(sys.path)

import local_exptools2

from local_exptools2.core import Session
from local_exptools2.core import Trial
from psychopy.visual import TextStim
from psychopy import visual
from local_exptools2 import utils, stimuli, physics, analysis, exp_design
import os
import pandas as pd



# from local_exptools2.stimuli import create_circle_fixation, create_cross_fixation, create_occluder, create_interactor     

class BehavTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None, trial_params=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, trial_params, **kwargs)
        self.txt = visual.TextStim(self.session.win, txt) 
        self.trial_params = trial_params
        self.trial_nr = trial_nr

    # def draw(self):
    #     """ Draws stimuli """
    #     [border.draw() for border in self.session.screen_borders] # Draw screen borders

    #     if self.phase == 0:
    #         self.txt.draw()
    #     else:
    #         self.session.occluder.draw() # Draw occluder
    #         [fix_line.draw() for fix_line in self.session.cross_fix] # Draw fixation cross
    #         if self.trial_params["trial_option"][:4] != "none": # Draw interactor if applicable
    #             stimuli.create_interactor(self.session.win, 
    #                                       self.trial_params["trial_option"],
    #                                       self.session.settings["ball"]["radius"],
    #                                       **self.session.settings["interactor"]).draw()
    
    def run(self, components, exp_data):
        """Run a single trial of the experiment."""
        config = components["config"]
        win = components["win"]
        verbose = config.experiment.verbose
        this_ballmov_time = None
        
        if verbose:
            print(f"Trial number: {self.trial_nr + 1}")
        
        # Initialize trial clock
        # trial_clock = core.Clock()
        trial_clock = self.session.clock
        # trial_clock = self.session.timer
        
        # Set up trial parameters
        trial_data = exp_design.setup_trial(components, self.trial_params, self.trial_nr)
        
        # Calculate total trial duration
        trial_duration = exp_design.calculate_trial_duration(components) # seems to work
        print(f"Trial duration: {trial_duration} seconds")
        # Store initial trial data
        exp_data = exp_design.store_trial_data(exp_data, components, self.trial_params, trial_data, self.trial_nr)
        
        # Run each phase of the trial
        exp_design.run_fixation_phase(components, trial_data, trial_clock, win, config)
        exp_design.run_interactor_phase(components, trial_data, trial_clock, win, config)
        exp_design.run_occluder_phase(components, trial_data, trial_clock, win, config)
        
        # Set up initial ball state
        velocity = exp_design.setup_ball_initial_state(components, trial_data)
        
        if verbose:
            print(f"Exact ballmovstart time: {trial_clock.getTime()}s")
        
        # BALL MOVEMENT LOOP
        # while trial_clock.getTime() < trial_duration: # I CHANGED THIS, perhaps the timing is fixed in the trial.py parent class
        # while trial_clock.getTime() < 
        if this_ballmov_time is None:
            this_ballmov_time = trial_clock.getTime() - (config.timing.fixation_dur + config.timing.interactor_dur + config.timing.occluder_dur)

        # Process ball movement for one frame
        velocity, ball_direction, exp_data, trial_data, _ = exp_design.process_ball_movement(
            components, trial_data, velocity, trial_clock, trial_duration, exp_data, this_ballmov_time
        )
        this_ballmov_time += components["frameDur"]
        # Handle user responses
        exp_data, trial_data = exp_design.handle_responses(
            components, trial_data, trial_clock, exp_data
        )
    
        # Calculate predictions for this trial
        exp_data = exp_design.calculate_predictions(trial_data, exp_data)
        
        return exp_data
                

        
class BehavSession(Session):
    """Behaviour session for physics prediction experiment."""    
    
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10):
        """ Initializes BehavSession object. """
        self.n_trials = n_trials
        self.settings_file = settings_file

        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)
        
        self.design_matrix = utils.build_design_matrix(
            n_trials = n_trials,
            change_ratio=[True],
            ball_color_change_mean=self.settings["ball"]["color_change_mean"], 
            ball_color_change_sd=self.settings["ball"]["color_change_sd"],
            verbose=self.settings["experiment"]["verbose"],
            neg_bias_factor=self.settings["ball"]["neg_bias_factor"],
        
        )
        
        print(self.design_matrix)

    def create_trials(self, durations=(.5, .5), timing='seconds'):
        """Run the entire experiment."""
        self.trials = []

        # Set up experiment components
        self.components = exp_design.setup_experiment(exp_window=self.win ,config_filename=self.settings_file)
        
        # Create design matrix and trial parameters
        trial_params = exp_design.create_design(self.components, verbose=True, n_trials=self.n_trials)
        
        # Initialize data structure
        self.exp_data = exp_design.initialize_data_structure(self.components, trial_params)
        
        # Show instructions
        exp_design.show_instructions(self.components)
        
        # Start time tracking
        # start_time = time.time()
        
        # MAIN EXPERIMENTAL LOOP
        # for trial_number in range(len(trial_params["trials"])):
        for trial_number in range(self.n_trials):
            print(f"Created trial {trial_number + 1} of {self.n_trials}")
            # Run a single trial
            # exp_data = exp_design.run_trial(components, trial_params, exp_data, trial_number)
            this_trial = BehavTrial(session=self,
                                        trial_nr=trial_number,
                                        phase_durations=durations,
                                        txt='Trial numero %i' % trial_number,
                                        parameters=dict(trial_type='even' if trial_number % 2 == 0 else 'odd'),
                                        verbose=True,
                                        timing=timing,
                                        trial_params=trial_params)
                                        # trial_params=self.design_matrix.iloc[trial_number],)
            
            self.trials.append(this_trial)
            
            # exp_data = this_trial.run(self.components, self.exp_data)
            
            # Show feedback at specified intervals
            # exp_design.show_feedback(self.components, trial_number, self.exp_data)
        
        # Close the window and save all data
        # components["win"].close() # This messes up shit
        
        # # Create final dataframe and save results # Not necessary now, but keep for later
        # df = pd.DataFrame(exp_data)
        # expInfo = components["expInfo"]
        # config = components["config"]
        # subject_id = expInfo["participant"]
        # task_name = expInfo["task"].lower().replace(" ", "_")
        # datadir = config.paths.datadir
        
        # exp_design.save_performance_data(expInfo["participant"], task_name, df, base_dir=datadir)
        # exp_design.save_performance_data(expInfo["participant"], task_name, trial_params["design_matrix"], design_matrix=True, base_dir=datadir)
        
        # Record timing information
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # timing_df = pd.DataFrame({"n_trials": [config.experiment.n_trials], "time_elapsed": [elapsed_time]})
        # timing_df.to_csv(f"{datadir}/{subject_id}/timing.csv")
        
        # return df
        
    # def create_trials(self, durations=(.5, .5), timing='seconds'):
    #     self.trials = []
    #     for trial_nr in range(self.n_trials):
            
    #         self.trials.append(
    #             BehavTrial(session=self,
    #                       trial_nr=trial_nr,
    #                       phase_durations=durations,
    #                       txt='Trial numero %i' % trial_nr,
    #                       parameters=dict(trial_type='even' if trial_nr % 2 == 0 else 'odd'),
    #                       verbose=True,
    #                       timing=timing,
    #                       trial_params=self.design_matrix.iloc[trial_nr],)
    #         )

    def run(self):
        """ Runs experiment. """
        self.start_experiment()
        if self.first_trial:
            # Show instructions
            exp_design.show_instructions(self.components)
            this_trial = 0
        
        for trial in self.trials:
            self.exp_data = trial.run(self.components, self.exp_data)            
            exp_design.show_feedback(self.components, trial_number=this_trial, exp_data=self.exp_data)
        this_trial += 1
        self.close()
        
        
        
        
if __name__ == '__main__': # This is so that it doesn't run when imported as a module

    # settings = op.join(op.dirname(__file__), 'behav_settings.yml')
    settings = op.join(op.abspath(op.join(op.dirname(__file__), '..')), 'behav_settings.yml')
    session = BehavSession('sub-07', 
                           n_trials=20, 
                           output_dir=op.join(os.getcwd(), "logs/wip"),
                           settings_file=settings)
    session.create_trials(durations=(10.5, 5.5), # Doesn't matter, it appears
                          timing='seconds')
    #session.create_trials(durations=(3, 3), timing='frames')
    session.run()
    session.quit()
    
    
