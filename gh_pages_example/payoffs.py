# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/Payoffs/02_payoffs1.ipynb.

# %% auto 0
__all__ = ['valid_dtypes', 'build_DSAIR', 'payoffs_sr', 'payoffs_sr_pfo_extension', 'payoffs_lr', 'punished_and_sanctioned_payoffs', 'payoffs_lr_peer_punishment', 'payoffs_lr_peer_reward', 'payoffs_lr_voluntary', 'payoffs_encanacao_2016', 'build_payoffs', 'compute_transition', 'state_transition', 'build_state_transitions', 'build_strategy', 'build_strategies']

# %% ../nbs/Payoffs/02_payoffs1.ipynb 2
from nbdev.showdoc import *
from fastcore.test import test_eq, test_close
from .model_utils import *
from .utils import *
from .types import *
import typing

import numpy as np
import nptyping

# %% ../nbs/Payoffs/02_payoffs1.ipynb 7
valid_dtypes = typing.Union[float, list[float], np.ndarray, dict]
def build_DSAIR(b:valid_dtypes=4, # benefit: The size of the per round benefit of leading the AI development race, b>0
                c:valid_dtypes=1, # cost: The cost of implementing safety recommendations per round, c>0
                s:valid_dtypes={"start":1, # speed: The speed advantage from choosing to ignore safety recommendations, s>1
                                "stop":5.1,
                                "step":0.1}, 
                p:valid_dtypes={"start":0, # avoid_risk: The probability that unsafe firms avoid an AI disaster, p ∈ [0, 1]
                                "stop":1.02,
                                "step":0.02}, 
                B:valid_dtypes=10**4, # prize: The size of the prize from winning the AI development race, B>>b
                W:valid_dtypes=100, # timeline: The anticipated timeline until the development race has a winner if everyone behaves safely, W ∈ [10, 10**6]
                pfo:valid_dtypes=0, # detection risk: The probability that firms who ignore safety precautions are found out, pfo ∈ [0, 1]
                α:valid_dtypes=0, # the cost of rewarding/punishing a peer
                γ:valid_dtypes=0, # the effect of a reward/punishment on a developer's speed
                epsilon:valid_dtypes=0, # commitment_cost: The cost of setting up and maintaining a voluntary commitment, ϵ > 0
                ω:valid_dtypes=0, # noise: Noise in arranging an agreement, with some probability they fail to succeed in making an agreement, ω ∈ [0, 1]
                collective_risk:valid_dtypes=0, # The likelihood that a disaster affects all actors
                β:valid_dtypes=0.01, # learning_rate: the rate at which players imitate each other
                Z:int=100, # population_size: the number of players in the evolutionary game
                strategy_set:list[str]=["AS", "AU"], # the set of available strategies
                exclude_args:list[str]=['Z', 'strategy_set'], # a list of arguments that should be returned as they are
                override:bool=False, # whether to build the grid if it is very large
                drop_args:list[str]=['override', 'exclude_args', 'drop_args'], # a list of arguments to drop from the final result
               ) -> dict: # A dictionary containing items from `ModelTypeDSAIR` and `ModelTypeEGT`
    """Initialise baseline DSAIR models for all combinations of the provided
    parameter valules. By default, we create models for replicating Figure 1
    of Han et al. 2021."""
    
    saved_args = locals()
    models = model_builder(saved_args,
                           exclude_args=exclude_args,
                           override=override,
                           drop_args=drop_args)
    return models

# %% ../nbs/Payoffs/02_payoffs1.ipynb 9
def payoffs_sr(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs_sr`
    """The short run payoffs for the DSAIR game."""
    s, b, c = [models[k] for k in ['s', 'b', 'c']]
    πAA = -c + b/2
    πAB = -c + b/(s+1)
    πBA = s*b/(s+1)
    πBB = b/2
    
    # Promote all stacks to 3D arrays
    πAA = πAA[:, None, None]
    πAB = πAB[:, None, None]
    πBA = πBA[:, None, None]
    πBB = πBB[:, None, None]
    matrix = np.block([[πAA, πAB], 
                       [πBA, πBB]])
    return {**models, 'payoffs_sr':matrix}

# %% ../nbs/Payoffs/02_payoffs1.ipynb 12
def payoffs_sr_pfo_extension(models):
    """The short run payoffs for the DSAIR game with a chance of unsafe
    behaviour being spotted."""
    s, b, c, pfo = [models[k] for k in ['s', 'b', 'c', 'pfo']]
    πAA = -c + b/2
    πAB = -c + b/(s+1) * (1 - pfo) + pfo * b
    πBA = (1 - pfo) * s * b / (s+1)
    πBB = (1 - pfo**2) * b/2
    
    # Promote all stacks to 3D arrays
    πAA = πAA[:, None, None]
    πAB = πAB[:, None, None]
    πBA = πBA[:, None, None]
    πBB = πBB[:, None, None]
    matrix = np.block([[πAA, πAB],
                       [πBA, πBB]])
    return {**models, 'payoffs_sr':matrix}

# %% ../nbs/Payoffs/02_payoffs1.ipynb 14
def payoffs_lr(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`
    """The long run average payoffs for the DSAIR game."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s, p, B, W = [models[k][:, None, None]
                  for k in ['s', 'p', 'B', 'W']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]    
    πAA = πAA + B/(2*W)
    πAB = πAB
    πBA = p*(s*B/W + πBA)
    πBB = p*(s*B/(2*W) + πBB)
    payoffs = np.block([[πAA, πAB],
                        [πBA, πBB]])
    return {**models, 'payoffs': payoffs}

# %% ../nbs/Payoffs/02_payoffs1.ipynb 16
def punished_and_sanctioned_payoffs(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
                                   ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`:
    """Compute the payoffs for the punished and sanctioner players in a DSAIR
    model with peer punishment."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s,b,c, p, B, W, pfo = [models[k][:, None, None]
                      for k in ['s', 'b', 'c', 'p', 'B', 'W', 'pfo']]
    α, γ = [models[k][:, None, None] for k in ['α', 'γ']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]
    
    s_punished = s - γ
    s_sanctioner = 1 - α
    sum_of_speeds = np.maximum(1e-20, s_punished + s_sanctioner)
    punished_wins = (s_punished > 0) & (((W-s)*np.maximum(0, s_sanctioner))
                                        <= ((W-1) * s_punished))
    punished_draws = (s_punished > 0) & (((W-s) * s_sanctioner)
                                         == ((W-1) * s_punished))
    sanctioner_wins = (s_sanctioner > 0) & (((W-s) * s_sanctioner)
                                            >= ((W-1)*np.maximum(0,s_punished)))
    no_winner = (s_punished <= 0) & (s_sanctioner <= 0)

    both_speeds_positive = (s_punished > 0) & (s_sanctioner > 0)
    only_sanctioner_speed_positive = (s_punished <= 0) & (s_sanctioner > 0)
    only_punisher_speed_positive = (s_punished > 0) & (s_sanctioner <= 0)

    p_loss = np.where(punished_wins | punished_draws, p, 1)
    R = np.where(no_winner,
                 1e50,
                 1 + np.minimum((W-s)/ np.maximum(s_punished, 1e-10),
                                (W-1)/ np.maximum(s_sanctioner, 1e-10)))
    B_s = np.where(sanctioner_wins, B, np.where(punished_draws, B/2, 0))
    B_p = np.where(punished_wins, B, np.where(punished_draws, B/2, 0))
    b_s = np.where(both_speeds_positive,
                   (1-pfo) * b * s_sanctioner / sum_of_speeds + pfo * b,
                   np.where(only_sanctioner_speed_positive, b, 0))
    b_p = np.where(both_speeds_positive,
                   (1-pfo) * b * s_punished / sum_of_speeds,
                   np.where(only_punisher_speed_positive, (1 - pfo)*b, 0))
    sanctioner_payoff = (1 / R) * (πAB + B_s - (b_s - c)) + (b_s - c)
    # sanctioner_payoff = (1 / R) * (πAB + B_s + (R-1)*(b_s - c))
    punished_payoff = (p_loss / R) * (πBA + B_p - b_p) + p_loss * b_p
    # punished_payoff = (p_loss / R) * (πBA + B_p + (R-1)*b_p)
    return {**models,
            'sanctioner_payoff':sanctioner_payoff,
            'punished_payoff':punished_payoff}

# %% ../nbs/Payoffs/02_payoffs1.ipynb 24
def payoffs_lr_peer_punishment(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`:
    """The long run average payoffs for the DSAIR game with peer punishment."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s,b,c, p, B, W = [models[k][:, None, None]
                      for k in ['s', 'b', 'c', 'p', 'B', 'W']]
    α, γ = [models[k][:, None, None] for k in ['α', 'γ']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]
    models = punished_and_sanctioned_payoffs(models)
    
    ΠAA = πAA + B/(2*W)
    ΠAB = πAB
    ΠAC = πAA + B/(2*W)
    ΠBA = p*(s*B/W + πBA)
    ΠBB = p*(s*B/(2*W) + πBB)
    ΠBC = models["punished_payoff"]
    ΠCA = πAA + B/(2*W)
    ΠCB = models["sanctioner_payoff"]
    ΠCC = πAA + B/(2*W)
    matrix = np.block([[ΠAA, ΠAB, ΠAC], 
                       [ΠBA, ΠBB, ΠBC],
                       [ΠCA, ΠCB, ΠCC],
                       ])
    return {**models, 'payoffs':matrix}

# %% ../nbs/Payoffs/02_payoffs1.ipynb 28
def payoffs_lr_peer_reward(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`:
    """The long run average payoffs for the DSAIR game with peer punishment."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s,b,c, p, B, W = [models[k][:, None, None]
                      for k in ['s', 'b', 'c', 'p', 'B', 'W']]
    α, γ = [models[k][:, None, None] for k in ['α', 'γ']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]
    
    s_rewarded = 1 + γ
    s_helper = np.maximum(0, 1 - α)
    s_colaborative = np.maximum(0, 1 + γ - α)
    ΠAA = πAA + B/(2*W)
    ΠAB = πBA
    ΠAC = πAA + B * s_rewarded / W
    ΠBA = p*(s*B/W + πBA)
    ΠBB = p*(s*B/(2*W) + πBB)
    ΠBC = p*(s*B/W + πBA)
    ΠCA = πAA
    ΠCB = πAB
    ΠCC = πAA + B * s_colaborative/(2*W)
    matrix = np.block([[ΠAA, ΠAB, ΠAC], 
                       [ΠBA, ΠBB, ΠBC],
                       [ΠCA, ΠCB, ΠCC],
                       ])
    return {**models, 'payoffs':matrix}

# %% ../nbs/Payoffs/02_payoffs1.ipynb 30
def payoffs_lr_voluntary(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`:
    """The long run average payoffs for the DSAIR game with voluntary
    commitments."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s,b,c, p, B, W = [models[k][:, None, None]
                      for k in ['s', 'b', 'c', 'p', 'B', 'W']]
    α, γ, ϵ = [models[k][:, None, None] for k in ['α', 'γ', 'epsilon']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]
    models = punished_and_sanctioned_payoffs(models)
    
    ΠAA = πAA + B/(2*W)
    ΠAB = πAB
    ΠAC = πAB
    ΠAD = πAB
    ΠAE = πAB
    ΠBA = p*(s*B/W + πBA)
    ΠBB = p*(s*B/(2*W) + πBB)
    ΠBC = p*(s*B/(2*W) + πBB)
    ΠBD = p*(s*B/(2*W) + πBB)
    ΠBE = p*(s*B/(2*W) + πBB)
    ΠCA = p*(s*B/W + πBA)
    ΠCB = p*(s*B/(2*W) + πBB)
    ΠCC = πAA + B/(2*W) - ϵ
    ΠCD = πAB - ϵ
    ΠCE = πAA + B/(2*W) - ϵ
    ΠDA = p*(s*B/W + πBA)
    ΠDB = p*(s*B/(2*W) + πBB)
    ΠDC = p*(s*B/W + πBA) - ϵ
    ΠDD = p*(s*B/(2*W) + πBB) - ϵ
    ΠDE = models['punished_payoff'] - ϵ
    ΠEA = p*(s*B/W + πBA) - ϵ
    ΠEB = p*(s*B/(2*W) + πBB)
    ΠEC = πAA + B/(2*W) - ϵ
    ΠED = models['sanctioner_payoff'] - ϵ
    ΠEE = πAA + B/(2*W) - ϵ
    matrix = np.block([[ΠAA, ΠAB, ΠAC, ΠAD, ΠAE], 
                       [ΠBA, ΠBB, ΠBC, ΠBD, ΠBE],
                       [ΠCA, ΠCB, ΠCC, ΠCD, ΠCE],
                       [ΠDA, ΠDB, ΠDC, ΠDD, ΠDE],
                       [ΠEA, ΠEB, ΠEC, ΠED, ΠEE]
                       ])
    return {**models, 'payoffs':matrix}

# %% ../nbs/Payoffs/03_payoffs2.ipynb 5
def payoffs_encanacao_2016(models):
    names = ['b_r', 'b_s', 'c_s', 'c_t', 'σ']
    b_r, b_s, c_s, c_t, σ = [models[k] for k in names]
    payoffs = {}
    n_players = 3
    n_sectors = 3
    n_strategies_per_sector = [2, 2, 2]
    n_strategies_total = 6
    # All players are from the first sector, playing that sector's first strategy
    index_min = "0-0-0"
    # All players are from the third sector, playing that sector's second strategy
    index_max = "5-5-5"
    # Note: The seperator makes it easy to represent games where n_strategies_total >= 10.

    # It is also trivial to define a vector which maps these indexes to strategy profiles
    # As sector order is fixed we could neglect to mention suscripts for each sector
    strategy_names = ["D", "C", "D", "C", "D", "C"]

    zero = np.zeros(b_r.shape[0])
    # As in the main text
    payoffs["C-C-C"] = {"P3": b_r-2*c_s,
                        "P2": σ+b_s-c_t,
                        "P1": σ+b_s}
    payoffs["C-C-D"] = {"P3": -c_s,
                        "P2": b_s-c_t,
                        "P1": zero}
    payoffs["C-D-C"] = {"P3": b_r-c_s,
                        "P2": zero,
                        "P1": b_s}
    payoffs["C-D-D"] = {"P3": zero,
                        "P2": σ,
                        "P1": σ}
    payoffs["D-C-C"] = {"P3": zero,
                        "P2": σ-c_t,
                        "P1": σ}
    payoffs["D-C-D"] = {"P3": zero,
                        "P2": -c_t,
                        "P1": zero}
    payoffs["D-D-C"] = {"P3": zero,
                        "P2": zero,
                        "P1": zero}
    payoffs["D-D-D"] = {"P3": zero,
                        "P2": σ,
                        "P1": σ}

    # The following indexes capture all strategy profiles where each player is fixed to a unique sector
    # (and player order does not matter, so we need only consider one ordering of sectors).
    payoffs["4-2-0"] = payoffs["D-D-D"]
    payoffs["4-2-1"] = payoffs["D-D-C"]
    payoffs["4-3-0"] = payoffs["D-C-D"]
    payoffs["4-3-1"] = payoffs["D-C-C"]
    payoffs["5-2-0"] = payoffs["C-D-D"]
    payoffs["5-2-1"] = payoffs["C-D-C"]
    payoffs["5-3-0"] = payoffs["C-C-D"]
    payoffs["5-3-1"] = payoffs["C-C-C"]
    return {**models, "payoffs": payoffs}


# %% ../nbs/Payoffs/03_payoffs2.ipynb 8
@multi
def build_payoffs(models: dict):
    return models.get('payoffs_key')


@method(build_payoffs, 'vasconcelos_2014_primitives')
def build_payoffs(models: dict):
    names = ['payoffs_state', 'c', 'T', 'b_r', 'b_p', 'r']
    payoffs_state, c, T, b_r, b_p, r = [models[k] for k in names]
    strategy_counts = payoffs_state['strategy_counts']
    n_r = strategy_counts["2"]
    n_p = strategy_counts["4"]
    risk = r * (n_r * c * b_r + n_p * c * b_p < T)
    # The payoffs must be computed for each strategy type in the interaction.
    # In games where we employ hypergeometric sampling, we usually do not
    # care about player order in the interaction. If order did matter, then
    # we would represent the payoffs per strategy still but it would capture
    # the expected payoffs given how likely a player of that strategy was to
    # play in each node of the extensive-form game. Non-players of type 0
    # usually do not have payoffs.
    payoffs = {"1": (1 - risk) * b_r,  # rich_free_rider
               "2": (1 - risk) * c * b_r,  # rich_contributor
               "3": (1 - risk) * b_p,  # poor_free_rider
               "4": (1 - risk) * c * b_p}  # poor_contributor
    return {**models, "payoff_primitives": payoffs}


@method(build_payoffs, 'vasconcelos_2014')
def build_payoffs(models: dict):
    profiles = create_profiles({'n_players': models.get('n_players', 5),
                                'n_strategies': [2, 2]})['profiles']
    payoffs = {}
    for profile in profiles:
        profile_tuple = thread_macro(profile,
                                     (str.split, "-"),
                                     (map, int, "self"),
                                     list,
                                     reversed,
                                     list,
                                     np.array,
                                     )
        strategy_counts = {f"{i}": np.sum(
            profile_tuple == i) for i in range(5)}
        payoffs_state = {'strategy_counts': strategy_counts}
        primitives = thread_macro(models,
                                  (assoc,
                                   'payoffs_state', payoffs_state,
                                   'payoffs_key', "vasconcelos_2014_primitives"),
                                  build_payoffs,
                                  (get, "payoff_primitives"),
                                  )
        payoffs[profile] = {}
        for i, strategy in enumerate(profile_tuple):
            if strategy == 0:
                continue
            elif strategy == 1:
                payoffs[profile][f"P{i+1}"] = primitives['1']
            elif strategy == 2:
                payoffs[profile][f"P{i+1}"] = primitives['2']
            elif strategy == 3:
                payoffs[profile][f"P{i+1}"] = primitives['3']
            elif strategy == 4:
                payoffs[profile][f"P{i+1}"] = primitives['4']
            else:
                continue
    return {**models, "payoffs": payoffs}


# %% ../nbs/Payoffs/03_payoffs2.ipynb 15
@method(build_payoffs, 'payoff_function_wrapper')
def build_payoffs(models: dict):
    profiles = create_profiles(models)['profiles']
    profile_payoffs_key = models['profile_payoffs_key']
    payoffs_state = models.get("payoffs_state", {})
    payoffs = {}
    for profile in profiles:
        profile_tuple = string_to_tuple(profile)
        strategy_counts = dict(zip(*np.unique(profile_tuple,
                                              return_counts=True)))
        payoffs_state = {**payoffs_state,
                         'strategy_counts': strategy_counts}
        profile_models = {**models,
                          "payoffs_state": payoffs_state,
                          "payoffs_key": profile_payoffs_key}
        profile_payoffs = thread_macro(profile_models,
                                       build_payoffs,
                                       (get, "profile_payoffs"),
                                       )
        payoffs[profile] = {}
        for i, strategy in enumerate(profile_tuple):
            if strategy == 0:
                # A strategy of 0 is reserved for missing players, missing
                # players do not have payoffs.
                continue
            elif str(strategy) in profile_payoffs.keys():
                payoffs[profile][f"P{i+1}"] = profile_payoffs[f"{strategy}"]
            else:
                continue
    return {**models, "payoffs": payoffs}


# %% ../nbs/Payoffs/03_payoffs2.ipynb 20
@method(build_payoffs, "flow_payoffs_wrapper")
def build_payoffs(models):
    "Build the flow payoffs for each state-action in a stochastic game."
    state_actions = models['state_actions']
    payoffs_state = models.get('payoffs_state', {})
    flow_payoffs = collections.defaultdict()
    for state_action in state_actions:
        state, action_profile = str.split(state_action, ":")
        action_tuple = string_to_tuple(action_profile)
        action_counts = dict(zip(*np.unique(action_tuple,
                                            return_counts=True)))
        payoffs_state = {**payoffs_state,
                         'strategy_counts': action_counts,
                         'state': state}
        payoffs_flow_key = models['payoffs_flow_key']
        profile_models = {**models,
                          "payoffs_state": payoffs_state,
                          "payoffs_key": payoffs_flow_key}
        flow_payoffs[state_action] = thread_macro(profile_models,
                                                  build_payoffs,
                                                  (get, "flow_payoffs"),
                                                  )
    return {**models, "flow_payoffs": flow_payoffs}


# %% ../nbs/Payoffs/03_payoffs2.ipynb 22
@multi
def compute_transition(models):
    "Compute the transition likelihood for the given transition."
    return models.get('compute_transition_key')

@method(compute_transition, 'anonymous_actions')
def compute_transition(models):
    """Compute transition likelihood when we are only passed anonymous action
    profiles (i.e. order does not matter)."""
    P, Q = [models[k] for k in ['P', 'Q']]
    transition_start, transition_end = [models[k] for k in ['transition_start',
                                                            'transition_end']]
    next_state, action_profile = transition_end.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    profiles = create_profiles({**models,
                                "profiles_rule": "from_strategy_count",
                                "strategy_count": action_count})['profiles']
    profile_tuples = map(string_to_tuple, profiles)
    p = [np.prod([P[f"P{player + 1}"][transition_start].get(f"A{action}", 0)
                  for player, action in enumerate(profile_tuple)])
         for profile_tuple in profile_tuples]
    return np.sum(p) * Q[transition_start][next_state]


@method(compute_transition)
def compute_transition(models):
    "Compute transition likelihood given the states and action profiles."
    P, Q = [models[k] for k in ['P', 'Q']]
    transition_start, transition_end = [models[k] for k in ['transition_start',
                                                            'transition_end']]
    next_state, action_profile = transition_end.split(":")
    action_tuple = string_to_tuple(action_profile)
    p = np.prod([P[f"P{player + 1}"][transition_start].get(f"A{action}", 0)
                 for player, action in enumerate(action_tuple)])
    return p * Q[transition_start][next_state]

# %% ../nbs/Payoffs/03_payoffs2.ipynb 23
@method(build_payoffs, "stochastic-no-discounting")
def build_payoffs(models: dict):
    """Compute the payoffs for a stochastic game with the given flow_payoffs,
    state_transitions, strategies, and strategy_profile, when there is no
    discounting."""
    u = models['flow_payoffs']
    Q = models['state_transitions']
    strategy_profile = models['strategy_profile'].split("-")[::-1]
    strategies = models['strategies']
    P = {f"P{player + 1}": strategies[strategy_key]
         for player, strategy_key in enumerate(strategy_profile)}
    state_actions = list(Q.keys())
    M = np.zeros((len(state_actions), len(state_actions)))
    for row, transition_start in enumerate(state_actions):
        for col, transition_end in enumerate(state_actions):
            transition_data = {**models,
                               "P": P,
                               "Q": Q,
                               "transition_start": transition_start,
                               "transition_end": transition_end}
            M[row, col] = compute_transition(transition_data)
    v = thread_macro({**models, "transition_matrix": np.array([M])},
                     find_ergodic_distribution,
                     (get, "ergodic"))[0]
    u = np.array([[u[s][f"{i+1}"] for i in range(len(u[s]))]
                  for s in state_actions])
    payoffs = np.dot(v, u)
    profile_payoffs = {f"{i+1}": pi for i, pi in enumerate(payoffs)}
    return {**models, "profile_payoffs": profile_payoffs}


@method(build_payoffs, "stochastic-with-discounting")
def build_payoffs(models: dict):
    """Compute the payoffs for a stochastic game with the given flow_payoffs,
    state_transitions, strategies, and strategy_profile."""
    u = models['flow_payoffs']
    Q = models['state_transitions']
    d = models['discount_rate']
    v0 = models['initial_state_action_distribution']
    strategy_profile = models['strategy_profile'].split("-")[::-1]
    strategies = models['strategies']
    P = {f"P{player + 1}": strategies[strategy_key]
         for player, strategy_key in enumerate(strategy_profile)}
    state_actions = list(Q.keys())
    M = np.zeros((len(state_actions), len(state_actions)))
    for row, transition_start in enumerate(state_actions):
        for col, transition_end in enumerate(state_actions):
            transition_data = {**models,
                               "P": P,
                               "Q": Q,
                               "transition_start": transition_start,
                               "transition_end": transition_end}
            M[row, col] = compute_transition(transition_data)
    v = (1 - d) * v0 * np.linalg.inv(np.eye(M.shape) - d * M)
    u = np.array([[u[s][f"{i+1}"] for i in range(len(u[s]))]
                  for s in state_actions])
    payoffs = np.dot(v, u)
    profile_payoffs = {f"{i+1}": pi for i, pi in enumerate(payoffs)}
    return {**models, "profile_payoffs": profile_payoffs}


# %% ../nbs/Payoffs/03_payoffs2.ipynb 29
@multi
def state_transition(models):
    "Compute the likelihood of the given state_transition."
    return models.get('state_transition_key')

@method(state_transition, 'ex1')
def state_transition(models):
    """Compute transition likelihood for a model with 2 states and an arbitrary
    number of players. To stay in the good state, 0, all players need to choose
    to cooperate, i.e. action 1."""
    state_action, next_state = [models[k] for k in ['state_action',
                                                    'next_state']]
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and next_state == '1'
        and n_cooperators != n_players):
        transition_likelihood = 1
    elif (current_state == '1'
          and next_state == '0'
          and n_cooperators == n_players):
        transition_likelihood = 1
    elif (current_state == '0'
          and next_state == '0'
          and n_cooperators == n_players):
        transition_likelihood = 1
    elif (current_state == '1'
          and next_state == '1'
          and n_cooperators != n_players):
        transition_likelihood = 1
    else:
        transition_likelihood = 0
    return transition_likelihood

# %% ../nbs/Payoffs/03_payoffs2.ipynb 30
def build_state_transitions(models):
    state_actions = models['state_actions']
    n_states = models['n_states']
    state_transitions = {}
    for state_action in state_actions:
        state_transitions[state_action] = {}
        for next_state in [f"{i}" for i in range(n_states)]:
            likelihood = state_transition({**models,
                                           "state_action": state_action,
                                           "next_state": next_state})
            state_transitions[state_action][next_state] = likelihood
    return {**models, "state_transitions": state_transitions}

# %% ../nbs/Payoffs/03_payoffs2.ipynb 35
@multi
def build_strategy(models):
    "Build the desired strategy"
    return models.get('strategy_key')

@method(build_strategy, 'ex1_rich_cooperator')
def build_strategy(models):
    """A rich player who cooperates with 95% probability if everyone currently
    cooperates, otherwise defects with 95% probability."""
    state_action = models['state_action']
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and n_cooperators == n_players):
        strategy = {"A1": 0.95, "A2": 0.05}
    elif (current_state == '0'
          and n_cooperators != n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    elif (current_state == '1'
          and n_cooperators == n_players):
        strategy = {"A1": 0.95, "A2": 0.05}
    elif (current_state == '1'
          and n_cooperators != n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    return strategy

@method(build_strategy, 'ex1_rich_defector')
def build_strategy(models):
    """A rich player who defects with 95% probability no matter what others
    do, nor what state they are in."""
    state_action = models['state_action']
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and n_cooperators == n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    elif (current_state == '0'
          and n_cooperators != n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    elif (current_state == '1'
          and n_cooperators == n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    elif (current_state == '1'
          and n_cooperators != n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    return strategy

@method(build_strategy, 'ex1_poor_cooperator')
def build_strategy(models):
    """A poor player who cooperates with 95% probability if everyone currently
    cooperates, otherwise defects with 95% probability."""
    state_action = models['state_action']
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and n_cooperators == n_players):
        strategy = {"A3": 0.95, "A4": 0.05}
    elif (current_state == '0'
          and n_cooperators != n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    elif (current_state == '1'
          and n_cooperators == n_players):
        strategy = {"A3": 0.95, "A4": 0.05}
    elif (current_state == '1'
          and n_cooperators != n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    return strategy

@method(build_strategy, 'ex1_poor_defector')
def build_strategy(models):
    """A poor player who defects with 95% probability no matter what others
    do, nor what state they are in."""
    state_action = models['state_action']
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and n_cooperators == n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    elif (current_state == '0'
          and n_cooperators != n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    elif (current_state == '1'
          and n_cooperators == n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    elif (current_state == '1'
          and n_cooperators != n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    return strategy

# %% ../nbs/Payoffs/03_payoffs2.ipynb 36
def build_strategies(models):
    "Build a dictionary containing the specified strategies in `models`"
    state_actions, strategy_keys = [models[k] for k in ["state_actions",
                                                        "strategy_keys"]]
    strategies = {f"{i+1}": {s: build_strategy({"strategy_key": strategy_key,
                                            "state_action": s})
                         for s in state_actions}
              for i, strategy_key in enumerate(strategy_keys)}
    return {**models, "strategies": strategies}