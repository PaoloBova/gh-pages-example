# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/models.ipynb.

# %% auto 0
__all__ = ['valid_dtypes', 'build_reg_market']

# %% ../nbs/models.ipynb 2
from .model_utils import *
from .types import *
from .utils import *

import typing

import fastcore.test
from nbdev.showdoc import *
import nptyping
import numpy as np

# %% ../nbs/models.ipynb 4
valid_dtypes = typing.Union[float, list[float], np.ndarray, dict]
def build_reg_market(b:valid_dtypes=4, # benefit: The size of the per round benefit of leading the AI development race, b>0
                c:valid_dtypes=1, # cost: The cost of implementing safety recommendations per round, c>0
                s:valid_dtypes={"start":1, # speed: The speed advantage from choosing to ignore safety recommendations, s>1
                                "stop":5.1,
                                "step":0.1}, 
                p:valid_dtypes={"start":0, # avoid_risk: The probability that unsafe firms avoid an AI disaster, p ∈ [0, 1]
                                "stop":1.02,
                                "step":0.02}, 
                B:valid_dtypes=10**4, # prize: The size of the prize from winning the AI development race, B>>b
                W:valid_dtypes=100, # timeline: The anticipated timeline until the development race has a winner if everyone behaves safely, W ∈ [10, 10**6]
                pfo_l:valid_dtypes=0, # detection_risk_lq: The probability that firms who ignore safety precautions are found out by high quality regulators, pfo_h ∈ [0, 1]
                pfo_h:valid_dtypes=0.5, # detection_risk_hq: The probability that firms who ignore safety precautions are found out by low quality regulators, pfo_h ∈ [0, 1]
                λ:valid_dtypes=0, # disaster_penalty: The penalty levied to regulators in case of a disaster
                r_l:valid_dtypes=0, # profit_lq: profits for low quality regulators before including government incentives, r_l ∈ R
                r_h:valid_dtypes=-1, # profit_hq: profits for high quality regulators before including government incentives, r_h ∈ R
                g:valid_dtypes=1, # government budget allocated to regulators per firm regulated, g > 0
                phi_h:valid_dtypes=0, # regulator_impact: how much do regulators punish unsafe firms they detect, by default detected firms always lose the race.
                phi2_h:valid_dtypes=0, # regulator_impact: how much do regulators punish 2 unsafe firms they detect, by default detected firms always lose the race.
                externality:valid_dtypes=0, # externality: damage caused to society when an AI disaster occurs
                decisiveness:valid_dtypes=1, # How decisive the race is after a regulator gets involved
                incentive_mix:valid_dtypes=1, # Composition of incentives, by default government always pays regulator but rescinds payment if unsafe company discovered
                collective_risk:valid_dtypes=0, # The likelihood that a disaster affects all actors
                β:valid_dtypes=1, # learning_rate: the rate at which players imitate each other
                Z:dict={"S1": 50, "S2": 50}, # population_size: the number of firms and regulators
                strategy_set:list[str]=["HQ-AS", "HQ-AU", "HQ-VS",
                                        "LQ-AS", "LQ-AU", "LQ-VS"], # the set of strategy combinations across all sectors
                exclude_args:list[str]=['Z', 'strategy_set'], # a list of arguments that should be returned as they are
                override:bool=False, # whether to build the grid if it is very large
                drop_args:list[str]=['override', 'exclude_args', 'drop_args'], # a list of arguments to drop from the final result
               ) -> dict: # A dictionary containing items from `ModelTypeRegMarket` and `ModelTypeEGT`
    """Initialise Regulatory Market models for all combinations of the provided
    parameter valules."""
    
    saved_args = locals()
    models = model_builder(saved_args,
                           exclude_args=exclude_args,
                           override=override,
                           drop_args=drop_args)
    return models