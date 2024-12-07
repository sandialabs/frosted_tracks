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

"""Wrappers for running TICC on trajectories given a set of feature functions

Functions:
- ticc_label_trajectory() - Identify point-by-point behavior for a single
  trajectory
- ticc_label_trajectories() - Identify point-by-point behavior jointly
  for all the points of a list of trajectories

"""

import psutil
from typing import List, Union

from fast_ticc import (
    ticc_labels,
    ticc_joint_labels,
    SingleDataSeriesResult,
    MultipleDataSeriesResult
)

import numba

from frosted_tracks.frosted_tracks_types import Trajectory
from frosted_tracks.feature_functions import trajectory_features
from frosted_tracks.segmentation import fast_cluster_predictor



def ticc_label_trajectory(trajectory: Trajectory,
                          feature_names: List[str],
                          ticc_num_labels: Union[str, int],
                          num_processors: int=32,
                          **solver_parameters) -> SingleDataSeriesResult:
    """Use TICC to compute labels for points in a trajectory.

    All keyword arguments apart from max_num_processors will be passed
    to the TICC solver.  This is how you set its hyperparameters.

    Arguments:
        trajectory (Tracktable trajectory): Input trajectory to label.
        feature_names (list of strings): Names of feature functions to use
            to generate data for this trajectory.
        ticc_num_labels (int or str): Number of labels to use for TICC.
            You can either supply an integer or the string "estimate" if
            you want to use the built-in (highly experimental) predictor.


    Keyword Arguments:
        num_processors (int): Maximum number of processors to let the TICC
            solver use.  Defaults to 32.  The solver will use this many
            processor cores or one core per cluster, whichever is less.
            Note that NumPy may also introduce its own multithreading that
            is not controlled by this parameter.

    Returns:
        TICC Result class for specified trajectory

    Raises:
        ValueError: ticc_num_labels is <= 0 or a string that is not "estimate"
    """



    if ((not isinstance(ticc_num_labels, (str, int)))
        or (isinstance(ticc_num_labels, str) and ticc_num_labels != "estimate")
        or (isinstance(ticc_num_labels, int) and ticc_num_labels <= 0)):
        raise ValueError("ticc_num_labels must either be a positive integer or the string 'estimate'.")

    cpu_count = min(psutil.cpu_count(), num_processors)
    numba.set_num_threads(cpu_count)

    data_series = trajectory_features(trajectory, feature_names)

    if ticc_num_labels == "estimate":
        ticc_num_labels = fast_cluster_predictor.fast_cluster_predictor(data_series)

    result = ticc_labels(data_series, num_processors=cpu_count,
                         num_clusters=ticc_num_labels,
                         **solver_parameters)

    return result



def ticc_label_trajectories(
        trajectories: List[Trajectory],
        feature_names: List[str],
        ticc_num_labels: Union[str, int],
        num_processors: int=4,
        **solver_parameters
    ) -> MultipleDataSeriesResult:
    """Run TICC on many trajectories at once.

    All keyword arguments besides num_processors will be passed to
    the TICC solver.  This is how you set its hyperparameters.

    Arguments:
        trajectories (list of Tracktable trajectories): Input data to label.
        feature_names (list of strings): Names of feature functions to use to
            generate data for each trajectory.
       ticc_num_labels (int or str): Number of labels to use for TICC.
            You can either supply an integer or the string "estimate" if
            you want to use the built-in (highly experimental) predictor.

    Keyword Arguments:
        num_processors (int): Maximum number of processors to let the TICC
            solver use.  Defaults to 32.  The solver will use this many
            processor cores or one core per cluster, whichever is less.
            Note that NumPy may also introduce its own multithreading that
            is not controlled by this parameter.

    Returns:
        List of TICC Result classes corresponding to input trajectories.  Note
        that the label switching cost is the same in each individual result and
        specifies the optimizer result across all trajectories.  We do not yet
        support breaking this out by each individual trajectory.

    Raises:
        ValueError: ticc_num_labels is <= 0 or a string that is not "estimate"

    """

    if ((not isinstance(ticc_num_labels, (str, int)))
        or (isinstance(ticc_num_labels, str) and ticc_num_labels != "estimate")
        or (isinstance(ticc_num_labels, int) and ticc_num_labels <= 0)):
        raise ValueError("ticc_num_labels must either be a positive integer or the string 'estimate'.")

    cpu_count = min(psutil.cpu_count(), num_processors)
    numba.set_num_threads(cpu_count)

    all_data_series = [
        trajectory_features(t, feature_names) for t in trajectories
        ]

    if ticc_num_labels == "estimate":
        ticc_num_labels = fast_cluster_predictor.fast_cluster_predictor_multiple_series(all_data_series)

    results = ticc_joint_labels(all_data_series,
                                num_Clusters=ticc_num_labels,
                                num_processors=cpu_count,
                                **solver_parameters)

    return results