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

"""Building blocks for PyTest test fixtures for trajectory data"""

import pytest

import datetime

import tracktable.examples.tutorials.tutorial_helper as tracktable_tutorial_helper
import frosted_tracks.trajectory

from frosted_tracks import feature_functions

from frosted_tracks.frosted_tracks_types import List, Trajectory

def _resample_trajectory(trajectory: Trajectory) -> Trajectory:
    return frosted_tracks.trajectory.resample_by_time(trajectory, resampling_interval())


@pytest.fixture
def resampling_interval() -> datetime.timedelta:
    return datetime.timedelta(seconds=10)


@pytest.fixture
def single_sample_trajectory() -> Trajectory:
    source_trajectories = tracktable_tutorial_helper.get_trajectory_list()
    # There's nothing special about this trajectory.  I just had to pick one.
    favorites = [t for t in source_trajectories if t[0].object_id == "367782880"]
    return favorites[0]


@pytest.fixture
def multiple_sample_trajectories() -> List[Trajectory]:
    return tracktable_tutorial_helper.get_trajectory_list()[0:10]


@pytest.fixture
def single_resampled_trajectory() -> Trajectory:
    return _resample_trajectory(single_sample_trajectory())


@pytest.fixture
def multiple_resampled_trajectories() -> List[Trajectory]:
    return [_resample_trajectory(t) for t in multiple_sample_trajectories()]


@pytest.fixture
def single_trajectory_features(simple_sensors):
    trajectory = single_resampled_trajectory()
    return frosted_tracks.feature_functions.trajectory_features(
        trajectory, simple_sensors)


@pytest.fixture
def multiple_trajectory_features(multiple_resampled_trajectories,
                                 simple_sensors):
    trajectories = multiple_resampled_trajectories
    result = []
    for t in trajectories:
        result.append(
            frosted_tracks.feature_functions.trajectory_features(
                t, simple_sensors
            )
        )
    return result


@pytest.fixture
def single_trajectory_raw_features(single_resampled_trajectory,
                                   simple_sensors):
    trajectory = single_resampled_trajectory
    sensors = simple_sensors

    result = {}
    for sensor in sensors:
        sensor_fn = feature_functions.feature_function(sensor)
        result[sensor] = sensor_fn(trajectory)
    return result

