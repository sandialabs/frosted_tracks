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

"""Internal methods for linear resampling of trajectories"""

import datetime
import math
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import scipy.interpolate

import tracktable.core.geomath

from frosted_tracks.frosted_tracks_types import (
    Trajectory, TrajectoryPoint,
    TerrestrialTrajectory, TerrestrialTrajectoryPoint,
    Cartesian3DTrajectory, Cartesian3DTrajectoryPoint
)

def resample_by_time_linear(
            trajectory: Trajectory,
            interval: datetime.timedelta,
            include_last_point: bool=False
            ) -> Trajectory:
    """Resample points along a trajectory to be evenly spaced in time

    This is an internal method.  You should not call it directly.
    Instead, use frosted_tracks.trajectory.resample_by_time().

    This function implements linear resampling.

    Arguments:
        trajectory (Tracktable trajectory): Trajectory to resample
        interval (datetime.timedelta): Desired spacing between points

    Keyword Arguments:
        include_last_point (bool): If true, the last point in the trajectory
        will be included in the resampled data even if it does not fall on the
        desired spacing.

    Returns:
        New trajectory with same type as original input
    """

    total_duration = trajectory[-1].timestamp - trajectory[0].timestamp
    new_trajectory_length = int(math.floor(total_duration / interval)) + 1
    increment = interval / total_duration

    return _resample_linear(trajectory, increment, new_trajectory_length,
                            tracktable.core.geomath.point_at_time_fraction,
                            include_last_point=include_last_point)


def resample_by_distance(trajectory: Trajectory, interval: float,
                         include_last_point: bool=False) -> Trajectory:
    """Resample points along a trajectory to be evenly spaced in distance

    Because TICC operates over a window of points, we need to guarantee
    that a window containing W points will cover the same amount of
    semantic ground no matter where the starting point is in the source
    trajectory.  There are two obvious ways to accomplish this: make sure
    the points are evenly spaced in time or in space.

    This function resamples by distance traveled.  If the last point in the
    trajectory does not fall exactly on the desired spacing, it will be omitted
    unless include_last_point is True.  We do this to avoid interpolating past
    the end of valid data.

    Arguments:
        trajectory (Tracktable trajectory): Trajectory to resample
        interval (float): Desired distance between points

    Keyword Arguments:
        include_last_point (bool): If true, the last point in the trajectory
        will be included in the resampled data even if it does not fall on the
        desired spacing.

    Returns:
        New trajectory with same type as original input
    """

    total_distance = trajectory[-1].current_length
    new_trajectory_length = int(math.floor(total_distance / interval)) + 1
    increment = interval / total_distance

    return _resample_linear(trajectory, new_trajectory_length, increment,
                     tracktable.core.geomath.point_at_length_fraction,
                     include_last_point=include_last_point)


def _copy_properties(source: Trajectory, destination: Trajectory):
    """Internal method: copy trajectory properties from T1 to T2"""
    for (name, value) in source.properties.items():
        destination.properties[name] = value


def _resample_linear(trajectory: Trajectory,
                     increment: float,
                     num_points: int,
                     sampler: Callable[[Trajectory, float], TrajectoryPoint],
                     include_last_point: bool=False) -> Trajectory:
    """Internal method: resample a trajectory

    Both resample_by_time and resample_by_distance dispatch to this
    function.  The difference is in the sampling function they
    supply to pull points out of the source trajectory.
    """
    new_trajectory = trajectory.__class__()
    _copy_properties(trajectory, new_trajectory)

    for i in range(num_points):
        new_trajectory.append(sampler(trajectory, i * increment))

    if include_last_point:
        if increment * num_points != 1.0:
            new_trajectory.append(trajectory[-1])

    return new_trajectory
