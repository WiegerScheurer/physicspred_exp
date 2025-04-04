import pickle
import random
import colour
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import truncexpon

def save_experiment(session, output_str, engine='pickle'):
    """ Saves Session object.

    parameters
    ----------
    session : Session instance
        Object created with Session class
    output_str : str
        name of output file (saves to current cwd) or complete filepath
    engine : str (default = 'pickle')
        Select engine to save object, either 'pickle' or 'joblib'
    """

    if engine == 'pickle':
        with open(output_str + '.pkl', 'w') as f_out:
            pickle.dump(session, f_out)
    else:
        raise ValueError("Engine not recognized, use 'pickle'")

def get_bounce_dist(ball_radius):
    """Compute the horizontal/vertical distance from ball center to 
    contactpoint of the interactor. These are the two right sides of 
    the triangle, where the diagonal is the ball radius.
    N.B.: As the triangle is an isosceles (gelijkbenig) triangle, the two 
    sides are equal.
    N.B.: As the angle is 45 degrees, we can just use the eenheidscirkel
    coordinates for 45 degrees = (sqrt(2) / 2)
    

    Args:
        ball_radius (float): radius of the ball

    Returns:
        float: the contactpoint coordinates between ball and interactor.
    """        
    return ball_radius * (np.sqrt(2) / 2)

def bellshape_sample(mean, sd, n_samples, plot:bool=False, shuffle:bool=True):
    
    sample_pool = np.array([random.normalvariate(mean, sd) for sample in range(n_samples)])
    
    if shuffle:
        random.shuffle(sample_pool)
    else:
        sample_pool.sort()
    if plot:
        plt.hist(sample_pool, bins=50)
    
    return list(sample_pool)

# Use for the different target hues, balance over it.
def ordinal_sample(mean, step_size, n_elements, plot:bool=False, round_decimals:int | None=None, 
                   pos_bias_factor:float=1.0, neg_bias_factor:float=1.0):
    # Calculate the start and end points
    half_range = (n_elements - 1) // 2
    start = mean - half_range * step_size
    end = mean + half_range * step_size
    
    # Generate the steps
    steps = np.arange(start, end + step_size, step_size)
    
    # Ensure the correct number of elements
    if len(steps) > n_elements:
        steps = steps[:n_elements]
        
    # Round the steps to 2 decimal places
    if round_decimals is not None:
        steps = np.round(steps, round_decimals)
        
    # Apply positive and negative bias factors
    if pos_bias_factor != 1.0:
        steps = [step * pos_bias_factor if step > mean else step for step in steps]
    if neg_bias_factor != 1.0:
        steps = [step * neg_bias_factor if step < mean else step for step in steps]
    
    return steps

# Compound function to be used in psychopy (make sure this is also usable in other projects, as quite important)
def oklab_to_rgb(oklab, psychopy_rgb:bool=False):
    # Convert OKLab to XYZ
    xyz = colour.Oklab_to_XYZ(oklab)
    # Convert XYZ to RGB
    rgb = [np.clip(((rgb_idx * 2) - 1), -1, 1) for rgb_idx in colour.XYZ_to_sRGB(xyz)] if psychopy_rgb else colour.XYZ_to_sRGB(xyz)

    return rgb

def create_balanced_trial_design(trial_n=None, change_ratio:list = [True, False], 
                                 ball_color_change_mean=0, ball_color_change_sd=0.05, startball_lum=.75, background_lum=.25,
                                 neg_bias_factor:float=1.5):
    
    def _clean_trial_options(df):
        # For each row, if trial_option starts with "none", keep only the first 6 characters
        df['trial_option'] = df['trial_option'].apply(
            lambda x: x[:6] if x.startswith('none_') else x
        )
        return df

    # Your options
    interactor_trial_options = ["45_top_r", "45_top_u", "45_bottom_l", "45_bottom_d",
                               "135_top_l", "135_top_u", "135_bottom_r", "135_bottom_d"]
    # empty_trial_options = ["none_l", "none_r", "none_u", "none_d"] * 2
    # Option 1: Create 8 truly unique empty trial options
    empty_trial_options = ["none_l_1", "none_r_1", "none_u_1", "none_d_1", 
                        "none_l_2", "none_r_2", "none_u_2", "none_d_2"]


    directions = ["left", "right"] * 8
    random.shuffle(directions)
    
    # Update the direction mapping
    direction_mapping = {
        "none_l_1": directions[0], "none_l_2": directions[1],
        "none_r_1": directions[2], "none_r_2": directions[3],
        "none_u_1": directions[4], "none_u_2": directions[5],
        "none_d_1": directions[6], "none_d_2": directions[7]
    }
    
    bounce_options = [True, False]
    # ball_change_options = [True, False]
    ball_change_options = change_ratio
    
    # Strangely enough it appears that darker balls should be less extreme than brighter balls. 
    
    ball_color_change_options = list(ordinal_sample(ball_color_change_mean, ball_color_change_sd, n_elements=5, round_decimals=3,
                                     pos_bias_factor=1.0, neg_bias_factor=neg_bias_factor))

    # If trial_n is specified, create a balanced subset
    if trial_n is not None:
        # Make sure trial_n is even for interactor:empty balance
        if trial_n % 2 == 1:
            trial_n -= 1
            print(f"Adjusted trial count to {trial_n} to maintain balance")
        
        half_n = trial_n // 2  # Half for interactor, half for empty
        
        # Create dataframe to store the balanced design
        all_trials = []
        
        # For interactor trials
        # First, create all possible combinations
        interactor_combos = list(product(
            interactor_trial_options,
            bounce_options,
            ball_change_options,
            ball_color_change_options
        ))
        random.shuffle(interactor_combos)  # Shuffle to avoid bias
        
        # Now intelligently select a subset that maximizes balance
        selected_interactor = []
        option_counts = {option: 0 for option in interactor_trial_options}
        bounce_counts = {True: 0, False: 0}
        change_counts = {True: 0, False: 0}
        luminance_counts = {luminance: 0 for luminance in ball_color_change_options}
        
        # First pass: try to get at least one of each option
        for option in interactor_trial_options:
            matching_combos = [c for c in interactor_combos if c[0] == option and c not in selected_interactor]
            if matching_combos:
                selected_interactor.append(matching_combos[0])
                option_counts[option] += 1
                bounce_counts[matching_combos[0][1]] += 1
                change_counts[matching_combos[0][2]] += 1
                luminance_counts[matching_combos[0][3]] += 1
        
        # Second pass: fill in remaining slots balancing bounce and ball_change
        remaining_slots = half_n - len(selected_interactor)
        while remaining_slots > 0:
            # Prioritize by least common option, then bounce, then ball_change
            min_option_count = min(option_counts.values())
            min_options = [opt for opt, count in option_counts.items() if count == min_option_count]
            
            min_bounce_count = min(bounce_counts.values())
            min_bounce = [b for b, count in bounce_counts.items() if count == min_bounce_count]
            
            min_change_count = min(change_counts.values())
            min_change = [c for c, count in change_counts.items() if count == min_change_count]
            
            # Find combos that match our criteria
            matching_combos = [c for c in interactor_combos 
                              if c[0] in min_options 
                              and c[1] in min_bounce 
                              and c[2] in min_change 
                              and c not in selected_interactor]
            
            # If no perfect match, relax constraints one by one
            if not matching_combos:
                matching_combos = [c for c in interactor_combos 
                                  if c[0] in min_options 
                                  and c[1] in min_bounce 
                                  and c not in selected_interactor]
            
            if not matching_combos:
                matching_combos = [c for c in interactor_combos 
                                  if c[0] in min_options 
                                  and c not in selected_interactor]
            
            if not matching_combos:
                matching_combos = [c for c in interactor_combos if c not in selected_interactor]
            
            if matching_combos:
                best_combo = matching_combos[0]
                selected_interactor.append(best_combo)
                option_counts[best_combo[0]] += 1
                bounce_counts[best_combo[1]] += 1
                change_counts[best_combo[2]] += 1
                luminance_counts[best_combo[3]] += 1
                remaining_slots -= 1
            else:
                # If we somehow run out of unique combinations
                break
        
        # Create the interactor trials from our selection
        for trial_option, bounce, ball_change, ball_luminance in selected_interactor:
            all_trials.append({
                'trial_type': 'interactor',
                'trial_option': trial_option,
                'bounce': bounce,
                'phant_bounce_direction': None,
                'ball_change': ball_change,
                'ball_luminance': ball_luminance
            })
        
        # For empty trials, use the same approach
        empty_combos = list(product(
            empty_trial_options,
            bounce_options,
            ball_change_options,
            ball_color_change_options
        ))
        random.shuffle(empty_combos)  # Shuffle to avoid bias
        
        # Now intelligently select a subset that maximizes balance
        selected_empty = []
        option_counts = {option: 0 for option in empty_trial_options}
        bounce_counts = {True: 0, False: 0}
        change_counts = {True: 0, False: 0}
        luminance_counts = {luminance: 0 for luminance in ball_color_change_options}
        
        # First pass: try to get at least one of each option
        for option in empty_trial_options:
            matching_combos = [c for c in empty_combos if c[0] == option and c not in selected_empty]
            if matching_combos:
                selected_empty.append(matching_combos[0])
                option_counts[option] += 1
                bounce_counts[matching_combos[0][1]] += 1
                change_counts[matching_combos[0][2]] += 1
                luminance_counts[matching_combos[0][3]] += 1
        
        # Second pass: fill in remaining slots balancing bounce and ball_change
        remaining_slots = half_n - len(selected_empty)
        while remaining_slots > 0:
            # Prioritize by least common option, then bounce, then ball_change
            min_option_count = min(option_counts.values())
            min_options = [opt for opt, count in option_counts.items() if count == min_option_count]
            
            min_bounce_count = min(bounce_counts.values())
            min_bounce = [b for b, count in bounce_counts.items() if count == min_bounce_count]
            
            min_change_count = min(change_counts.values())
            min_change = [c for c, count in change_counts.items() if count == min_change_count]
            
            # Find combos that match our criteria
            matching_combos = [c for c in empty_combos 
                              if c[0] in min_options 
                              and c[1] in min_bounce 
                              and c[2] in min_change 
                              and c not in selected_empty]
            
            # If no perfect match, relax constraints one by one
            if not matching_combos:
                matching_combos = [c for c in empty_combos 
                                  if c[0] in min_options 
                                  and c[1] in min_bounce 
                                  and c not in selected_empty]
            
            if not matching_combos:
                matching_combos = [c for c in empty_combos 
                                  if c[0] in min_options 
                                  and c not in selected_empty]
            
            if not matching_combos:
                matching_combos = [c for c in empty_combos if c not in selected_empty]
            
            if matching_combos:
                best_combo = matching_combos[0]
                selected_empty.append(best_combo)
                option_counts[best_combo[0]] += 1
                bounce_counts[best_combo[1]] += 1
                change_counts[best_combo[2]] += 1
                luminance_counts[best_combo[3]] += 1
                remaining_slots -= 1
            else:
                # If we somehow run out of unique combinations
                break
        
        # Create the empty trials from our selection
        for trial_option, bounce, ball_change, ball_luminance in selected_empty:
            bounce_direction = direction_mapping[trial_option] if bounce else None
            all_trials.append({
                'trial_type': 'empty',
                'trial_option': trial_option,
                'bounce': bounce,
                'phant_bounce_direction': bounce_direction,
                'ball_change': ball_change,
                'ball_luminance': ball_luminance
            })
        
        # Convert to dataframe and shuffle
        df = pd.DataFrame(all_trials)
        df.sample(frac=1).reset_index(drop=True)
        return _clean_trial_options(df)
    
    # If trial_n is None, create the full balanced design
    else:
        # Create all possible combinations
        all_trials = []
        
        # For interactor trials
        for combo in product(interactor_trial_options, bounce_options, ball_change_options, ball_color_change_options):
            trial_option, bounce, ball_change, ball_luminance = combo
            all_trials.append({
                'trial_type': 'interactor',
                'trial_option': trial_option,
                'bounce': bounce,
                'phant_bounce_direction': None,
                'ball_change': ball_change,
                'ball_luminance': ball_luminance
            })
        
        # For empty trials - we need to duplicate these to match interactor count
        for combo in product(empty_trial_options, bounce_options, ball_change_options, ball_color_change_options):
            trial_option, bounce, ball_change, ball_luminance = combo
            bounce_direction = direction_mapping[trial_option] if bounce else None
            
            # Each empty trial combination needs to appear twice to balance with interactor trials
            # for _ in range(2):
            all_trials.append({
                'trial_type': 'empty',
                'trial_option': trial_option,
                'bounce': bounce,
                'phant_bounce_direction': bounce_direction,
                'ball_change': ball_change,
                'ball_luminance': ball_luminance
            })
    
    
        # Convert to dataframe and shuffle
        df = pd.DataFrame(all_trials)

        df.sample(frac=1).reset_index(drop=True)
        
        return _clean_trial_options(df)

def build_design_matrix(n_trials:int, change_ratio:list=[True, False], 
                        ball_color_change_mean:float=.45, ball_color_change_sd:float=.05, 
                        trials_per_fullmx:int | None=None, verbose:bool=False,
                        neg_bias_factor:float=1.5):
    """
    Build a design matrix for a given number of trials.

    Parameters:
    - n_trials (int): The total number of trials.
    - verbose (bool): Whether to print verbose output.

    Returns:
    - design_matrix (pd.DataFrame): The resulting design matrix.
    """
    # trials_per_fullmx = 192
    if trials_per_fullmx is None:
        test_dm = create_balanced_trial_design(trial_n=None, 
                                               change_ratio=change_ratio, 
                                               ball_color_change_mean=ball_color_change_mean, 
                                               ball_color_change_sd=ball_color_change_sd,
                                               neg_bias_factor=neg_bias_factor)
        trials_per_fullmx = len(test_dm)    
        print(f"Number of trials per full matrix: {trials_per_fullmx}")

    full_matrices = n_trials // trials_per_fullmx
    remainder = n_trials % trials_per_fullmx
    
    if verbose:
        print(f"Design matrix for {n_trials} trials, constituting {full_matrices} fully balanced matrices and {remainder} trials balanced approximately optimal.")
    
    if remainder > 0:
        initial_dm = create_balanced_trial_design(remainder, change_ratio=change_ratio, 
                                                  ball_color_change_mean=ball_color_change_mean, 
                                                  ball_color_change_sd=ball_color_change_sd,
                                                  neg_bias_factor=neg_bias_factor)
    else:
        initial_dm = pd.DataFrame()
    
    for full_matrix in range(full_matrices + 1):
        dm = create_balanced_trial_design(192, neg_bias_factor=neg_bias_factor)
        dm = create_balanced_trial_design(trials_per_fullmx, change_ratio=change_ratio, 
                                          ball_color_change_mean=ball_color_change_mean, 
                                          ball_color_change_sd=ball_color_change_sd,
                                          neg_bias_factor=neg_bias_factor)
        if full_matrix == 0:
            design_matrix = initial_dm
        else:
            design_matrix = pd.concat([design_matrix, dm])
            
    # Shuffle the rows and reset the index
    design_matrix = design_matrix.sample(frac=1).reset_index(drop=True)
    return design_matrix