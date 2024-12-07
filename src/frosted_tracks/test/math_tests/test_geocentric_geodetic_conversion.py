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

"""Test coordinate conversion for Hermite interpolation"""

import os.path
import pytest

import tracktable.domain.terrestrial

from frosted_tracks.trajectory.resample.hermite import (
    _trajectory_geodetic_to_geocentric,
    _trajectory_geocentric_to_geodetic
)


@pytest.fixture
def mappy_trajectory(frosted_tracks_data_directory):
    # Data files are in data/ off the root of the repository;
    # we're in src/frosted_tracks/test
    filename = os.path.join(frosted_tracks_data_directory, "mappy_trajectories.traj")
    with open(filename, "rb") as infile:
        reader = tracktable.domain.terrestrial.TrajectoryReader(infile)
        all_trajectories = list(reader)
        return all_trajectories[0]


def test_geocentric_to_geodetic(mappy_trajectory):
    has_altitude = "altitude" in mappy_trajectory[0].properties
    if has_altitude:
        altitude_name = "altitude"
    else:
        altitude_name = None

    print("---- Converting geocentric to geodetic")
    geocentric = _trajectory_geodetic_to_geocentric(mappy_trajectory, altitude_name=altitude_name)

    print("---- Converting geodetic to geocentric")
    new_geodetic = _trajectory_geocentric_to_geodetic(geocentric, altitude_name=altitude_name)

    longitude_error = []
    latitude_error = []
    altitude_error = []

    for (old_point, new_point) in zip(mappy_trajectory, new_geodetic):
        longitude_error.append(new_point[0] - old_point[0])
        latitude_error.append(new_point[1] - old_point[1])
        if has_altitude:
            altitude_error.append(new_point.properties["altitude"] - old_point.properties["altitude"])
        else:
            altitude_error.append(0)

    avg_longitude_error = sum(longitude_error) / len(longitude_error)
    median_longitude_error = sorted(longitude_error)[int(len(longitude_error)/2)]

    avg_latitude_error = sum(latitude_error) / len(latitude_error)
    median_latitude_error = sorted(latitude_error)[int(len(latitude_error)/2)]

    avg_altitude_error = sum(altitude_error) / len(altitude_error)
    median_altitude_error = sorted(altitude_error)[int(len(altitude_error)/2)]

    print((
        f"Geocentric to geodetic: average errors are {avg_longitude_error} (longitude), "
        f"{avg_latitude_error} (latitude), {avg_altitude_error} (altitude)"))

    print((
        f"Geocentric to geodetic: median errors are {median_longitude_error} (longitude), "
        f"{median_latitude_error} (latitude), {median_altitude_error} (altitude)"))

    assert avg_longitude_error < 1e-4
    assert avg_latitude_error < 1e-4
