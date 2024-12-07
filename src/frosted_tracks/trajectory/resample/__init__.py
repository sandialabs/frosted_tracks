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

"""Resample a trajectory to have evenly spaced points in time

Functions:

- resample_by_time: Recompute points for a trajectory so that they are
  spaced evenly in time.  Includes options for linear interpolate, PCHIP
  and Hermite spline interpolation.

"""

import datetime
from frosted_tracks.frosted_tracks_types import Trajectory

from frosted_tracks.trajectory.resample.linear import resample_by_time_linear
from frosted_tracks.trajectory.resample.pchip import resample_by_time_pchip
from frosted_tracks.trajectory.resample.hermite import resample_by_time_hermite


__all__ = [
    "resample_by_time"
    ]


def resample_by_time(trajectory: Trajectory, interval: datetime.timedelta,
                     include_last_point: bool=False,
                     interpolation: str="linear",
                     speed_property: str="speed",
                     heading_property: str="heading") -> Trajectory:
    """Resample points along a trajectory to be evenly spaced in time

    Because TICC operates over a window of points, we need to guarantee
    that a window containing W points will cover the same amount of
    semantic ground no matter where the starting point is in the source
    trajectory.  There are two obvious ways to accomplish this: make sure
    the points are evenly spaced in time or in space.

    This function resamples by time.  If the last point in the trajectory
    does not fall exactly on the desired spacing, it will be omitted unless
    include_last_point is True.  We do this to avoid interpolating past the end
    of valid data.

    You can choose the type of interpolation for coordinates.  Valid
    values are "linear", "pchip", and "hermite".

    - Linear interpolation: Coordinates are sampled from the line segment
          between the nearest two points in the source trajectory.
          Only requires position and timestamp (which all trajectories
          have)
    - PCHIP interpolation: Coordinates are sampled from a piecewise cubic
          Hermite polynomial fit to the points of the original trajectory.
          This gives smoother results than linear interpolation.  See
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
          for more information.  Like linear interpolation, this only requires
          position and time.  The paths are smoother than the ones
          that come from linear interpolation.
    - Hermite interpolation: If you happen to have speed and heading
          values at all points on your trajectory (we often do!) then
          we can use that kinematic information in the interpolation.
          In that case, we use a piecewise cubic Hermite spline and
          force the derivatives to match the object's velocity vector
          at each point.

    Arguments:
        trajectory (Tracktable trajectory): Trajectory to resample
        interval (datetime.timedelta): Desired spacing between points

    Keyword Arguments:
        include_last_point (bool): If true, the last point in the trajectory
            will be included in the resampled data even if it does not fall on the
            desired spacing.  Defaults to False.
        interpolation (str): Type of interpolation to use.  Legal values are
            "linear", "pchip", and "hermite".  Defaults to "linear".
        speed_property (str): Name of speed property on points.  Defaults
            to "speed".  Only used when interpolation is "hermite".
        heading_property (str): Name of heading property on points.  Defaults
            to "heading".  Only used when interpolation is "hermite".

    Returns:
        New trajectory with same type as original input
    """

    if interpolation == "linear":
        return resample_by_time_linear(trajectory, interval,
                                       include_last_point=include_last_point)

    if interpolation == "pchip":
        return resample_by_time_pchip(trajectory, interval,
                                      include_last_point=include_last_point)

    if interpolation == "hermite":
        return resample_by_time_hermite(trajectory, interval,
                                        include_last_point=include_last_point,
                                        speed_name=speed_property,
                                        heading_name=heading_property)

    raise ValueError((
        "resample_by_time: Valid interpolation schemes are 'linear', "
        f"'pchip', and 'hermite'.  You supplied '{interpolation}'."
    ))
