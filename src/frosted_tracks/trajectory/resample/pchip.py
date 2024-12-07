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

"""Internal methods for PCHIP resampling of trajectory coordinates"""

import datetime

import frosted_tracks.trajectory.resample.linear
from frosted_tracks.frosted_tracks_types import TerrestrialTrajectory

import numpy as np
import scipy.interpolate

def _build_interpolators(data):
    """Build the objects that will resample the coordinates"""
    x_interpolator = scipy.interpolate.PchipInterpolator(data[0, :], data[1, :])
    y_interpolator = scipy.interpolate.PchipInterpolator(data[0, :], data[2, :])

    return (x_interpolator, y_interpolator)


def _resample_coordinates(x_interp, y_interp, t_values: np.ndarray) -> np.ndarray:
    """Actually resample X and Y at the specified T values"""
    num_points = t_values.shape[0]
    result = np.zeros(shape=(3, num_points))

    for i in range(num_points):
        t = t_values[i]
        result[0, i] = t
        result[1, i] = x_interp(t)
        result[2, i] = y_interp(t)

    return result

def _trajectory_to_xyt(trajectory: TerrestrialTrajectory) -> np.ndarray:
    """Internal method -- convert trajectory to an array

    The array has 3 rows: x coordinates, y coordinates, and
    elapsed seconds.
    """

    t = list()
    x = list()
    y = list()

    start_time = trajectory[0].timestamp
    end_time = trajectory[-1].timestamp
    duration = end_time - start_time

    for point in trajectory:
        t.append((point.timestamp - start_time) / duration)
        x.append(point[0])
        y.append(point[1])

    return np.array([t, x, y])



def resample_by_time_pchip(trajectory: TerrestrialTrajectory,
                           interval: datetime.timedelta,
                           include_last_point: bool=False) -> TerrestrialTrajectory:
    """Resample trajectory coordinates using PCHIP

    This function resamples an existing trajectory to create a version with
    points evenly spaced in time.  It uses PCHIP (CITE) to fit a smooth
    curve to the existing coordinates.

    All properties other than the coordinates are sampled with ordinary
    linear interpolation.

    Arguments:
        trajectory (Tracktable trajectory): Trajectory to resample
        interval (datetime.timedelta): Desired time between points

    Keyword Arguments:
        include_last_point (bool): If True, the last point in the original
            trajectory will be included whether or not it falls exactly on
            the desired time boundary.

    Returns:
        New trajectory with points at the requested intervals
    """



    # Resample at the appropriate interval to get interpolated property values
    new_trajectory = frosted_tracks.trajectory.resample.linear.resample_by_time_linear(
        trajectory, interval, include_last_point=include_last_point
        )

    # Build the spline interpolator
    xyt = _trajectory_to_xyt(trajectory)
    (x_interp, y_interp) = _build_interpolators(xyt)

    # The endpoints may have been adjusted slightly during
    # interpolation to avoid falling off the end
    new_start_time = new_trajectory[0].timestamp
    new_end_time = new_trajectory[-1].timestamp
    new_duration = new_end_time - new_start_time

    def t_fraction(point):
        return (point.timestamp - new_start_time) / new_duration

    new_t_values = np.array([t_fraction(point) for point in new_trajectory])
    new_coordinates = _resample_coordinates(x_interp, y_interp, new_t_values)
    object_id = new_trajectory[0].object_id
    for i in range(len(new_trajectory)):
        new_trajectory[i][0] = new_coordinates[1, i]
        new_trajectory[i][1] = new_coordinates[2, i]

    return new_trajectory
