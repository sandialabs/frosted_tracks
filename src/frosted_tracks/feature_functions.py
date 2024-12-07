# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the
# distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Register and retrieve feature functions.

A feature function computes a numeric value for each point in a
trajectory.  Feature functions are used to convert one or more
trajectories into data sets that can be labeled by TICC.
"""

__all__ = [
    "FeatureFunction",
    "feature_function",
    "list_feature_functions",
    "register_feature_function",
    "remove_feature_function",
    "trajectory_features"
    ]


from typing import List, Optional

import numpy as np

from frosted_tracks.frosted_tracks_types import FeatureFunction, Trajectory

# Registry where we will keep all feature functions

FEATURE_FUNCTIONS = {}


def register_feature_function(fn: FeatureFunction, name: Optional[str]=None) -> FeatureFunction:
    """Decorator: define and register a feature function by name

    A feature function takes a single parameter as input (a trajectory) and
    produces a sequence of floats with the same length as the trajectory.

    If you decorate your feature function with @register_feature_function,
    it will be saved by its name for easy lookup later with feature_function().
    By default, the name will be whatever name the function is given in its
    definition.  You can also supply an argument to the decorator to change
    the name.

    Example:

    @register_feature_function
    def speed(trajectory):
        return [point.speed for point in trajectory]

    ...or...

    @register_feature_function("speed_by_another_name")
    def speed(trajectory):
        return [point.speed for point in trajectory]

    This feature function creates a data series containing the value of the
    'speed' attribute at every point in the trajectory.  You can later refer
    to this function by asking for feature_function("speed") (in the first case)
    or feature_function("speed_by_another_name") (in the second).
    """

    global FEATURE_FUNCTIONS
    if name is None:
        name = fn.__name__
    if name in FEATURE_FUNCTIONS:
        print(f"WARNING: Replacing existing feature function {name}.")
    FEATURE_FUNCTIONS[name] = fn
    return fn


def feature_function(fn_name: str) -> FeatureFunction:
    """Retrieve a feature function by name.

    Given a string, this function will retrieve the feature function
    named by that string.

    Example:

    >>> @register_feature_function
    >>> def my_feature(trajectory):
    >>>   # your code goes here

    >>> ff = feature_function("my_feature")

    At this point, 'ff' will hold a pointer to the function "my_feature".

    Arguments:
        fn_name (str): Name of function to retrieve

    Returns:
        Pointer to feature function

    Raises:
        KeyError: No function was registered with that name
    """

    if fn_name not in FEATURE_FUNCTIONS:
        raise KeyError(f"There is no feature function '{fn_name}' currently registered.")
    return FEATURE_FUNCTIONS[fn_name]


def list_feature_functions() -> List[str]:
    """Report the names of all registered feature functions"""

    return list(FEATURE_FUNCTIONS.keys())


def remove_feature_function(name: str) -> None:
    """Remove a feature function from the registry

    This will remove a feature function from the registry but not
    delete the original binding.  For example:

    >>> @register_feature_function
    >>> def speed(t):
    >>>     return [point.speed for point in t]
    >>>
    >>> feature_function("speed")
    <function __main__.speed(t)>

    >>> remove_feature_function("speed")
    >>> feature_function("speed")

    KeyError: "There is no feature function 'speed' currently registered."

    However, you can still access speed() as a function in the scope
    where it was originally defined.

    Arguments:
        name (str): Name of function to unregister

    Returns:
        None

    Raises:
        KeyError: no feature function by the specified name exists
    """

    global FEATURE_FUNCTIONS
    if name not in FEATURE_FUNCTIONS:
        raise KeyError(f"There is no feature function '{name}' currently registered.")
    FEATURE_FUNCTIONS.pop(name)


def trajectory_features(trajectory: Trajectory, feature_names: List[str]) -> np.ndarray:
    """Construct a NumPy array with a set of named features

    This is a glue function.  It creates a NumPy array containing
    feature values for a single trajectory.  Each row contains the
    various sensor values at a single time point.  Each data series
    (sensor) is a column in the array.

    Columns will have the same order as the feature functions.

    Feature functions must be specified by name.  They will be looked up using
    frosted_tracks.feature_functions.feature_function().

    Arguments:
        trajectory (Tracktable trajectory): Trajectory to compute data for
        feature_names (list of str): Feature functions to look up

    Returns:
        New NumPy array with shape (num_features, len(trajectory))

    Raises:
        KeyError: Some feature function was requested that was not available
        IndexError: Some feature function returned the wrong number of values
    """

    data_values = [
        feature_function(name)(trajectory) for name in feature_names
    ]

    for (name, values) in zip(feature_names, data_values):
        if len(values) != len(trajectory):
            raise IndexError((
                f"Feature {name} should have returned {len(trajectory)} values "
                f"for a trajectory with {len(trajectory)} points but returned "
                f"{len(values)} instead."))

    return np.array(data_values).T

