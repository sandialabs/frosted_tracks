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

"""PyTest wrappers for all fixtures"""


import datetime

import pytest
import numpy as np

from frosted_tracks.test.fixtures import common
from frosted_tracks.test.fixtures import features
from frosted_tracks.test.fixtures import ticc
from frosted_tracks.test.fixtures import trajectories

from frosted_tracks.frosted_tracks_types import List, Trajectory

@pytest.fixture(scope="module")
def simple_sensors() -> List[str]:
    return features.simple_sensors()

@pytest.fixture
def resampling_interval() -> datetime.timedelta:
    return trajectories.resampling_interval()

@pytest.fixture
def ticc_window_size() -> int:
    return features.window_size()

@pytest.fixture
def single_sample_trajectory() -> Trajectory:
    return trajectories.single_sample_trajectory()

@pytest.fixture
def multiple_sample_trajectories() -> List[Trajectory]:
    return trajectories.multiple_sample_trajectories()

@pytest.fixture(scope="module")
def single_resampled_trajectory() -> Trajectory:
    return trajectories.single_resampled_trajectory()

@pytest.fixture(scope="module")
def multiple_resampled_trajectories() -> List[Trajectory]:
    return trajectories.multiple_resampled_trajectories()

@pytest.fixture(scope="module")
def single_trajectory_features() -> np.ndarray:
    return trajectories.single_trajectory_features()

@pytest.fixture(scope="module")
def multiple_trajectory_features() -> np.ndarray:
    return trajectories.multiple_trajectory_features()

@pytest.fixture
def single_trajectory_raw_features() -> dict:
    return trajectories.single_trajectory_raw_features()

@pytest.fixture(scope="module")
def ticc_label_switching_cost() -> float:
    return ticc.ticc_label_switching_cost()

@pytest.fixture(scope="module")
def ticc_num_clusters() -> int:
    return ticc.ticc_num_clusters()

@pytest.fixture(scope="module")
def ticc_window_size() -> int:
    return ticc.window_size()

@pytest.fixture(scope="module")
def random_seed() -> int:
    return common.random_seed()

@pytest.fixture(scope="module")
def golden_input():
    return common.golden_input()

@pytest.fixture(scope="module")
def min_meaningful_covariance() -> float:
    return ticc.min_meaningful_covariance()