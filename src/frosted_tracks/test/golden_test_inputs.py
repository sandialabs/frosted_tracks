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

### BE CAREFUL before renaming this file.  Its name affects the
### location where the golden test input data will be stored.
###
### If you change it, also change
### frosted_tracks.test.fixtures.common.golden_input_oracle() to match
### the new name.

import sys

from frosted_tracks.test.fixtures import features as feature_fixtures
from frosted_tracks.test.fixtures import ticc as ticc_fixtures
from frosted_tracks.test.fixtures import trajectories as trajectory_fixtures

from frosted_tracks.test.infrastructure.ground_truth import GroundTruthOracle
from frosted_tracks.segmentation.ticc import data_preparation
from frosted_tracks import feature_functions

"""Generate potentially-platform-varying inputs once for consistency

These are inputs that might conceivably differ from platform to
platform because of different implementations of trigonometric or
transcendental functions.
"""

def generate_ground_truth(oracle: GroundTruthOracle):
    single_trajectory_features = trajectory_fixtures.single_trajectory_features()
    single_trajectory_stacked_data = data_preparation.stack_training_data(
        single_trajectory_features, ticc_fixtures.window_size())
    oracle.put(single_trajectory_features, "single_trajectory_features", overwrite=True)
    oracle.put(single_trajectory_stacked_data, "single_trajectory_stacked_data", overwrite=True)

    multiple_trajectory_features = trajectory_fixtures.multiple_trajectory_features()
    multiple_trajectory_stacked_data = data_preparation.stack_training_data_multiple_series(
        multiple_trajectory_features, ticc_fixtures.window_size())
    oracle.put(multiple_trajectory_features, "multiple_trajectory_features", overwrite=True)
    oracle.put(multiple_trajectory_stacked_data, "multiple_trajectory_stacked_data", overwrite=True)

    single_trajectory_raw_features = feature_functions.trajectory_features(
        trajectory_fixtures.single_resampled_trajectory(),
        feature_fixtures.simple_sensors()
    )
    oracle.put(single_trajectory_raw_features, "single_trajectory_raw_features", overwrite=True)

def main():
    print("These are not the droids you're looking for.")
    return 0


if __name__ == '__main__':
    sys.exit(main())