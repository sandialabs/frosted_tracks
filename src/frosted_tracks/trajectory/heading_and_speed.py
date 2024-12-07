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

"""recompute_heading and recompute_speed for the trajectory module

These functions are described in the docstring for
frosted_tracks.trajectory.
"""

__all__ = ["recompute_heading", "recompute_speed"]

import tracktable.core.geomath

from frosted_tracks.frosted_tracks_types import TerrestrialTrajectory

def recompute_heading(trajectory: TerrestrialTrajectory,
                       property_name: str="heading") -> TerrestrialTrajectory:
    """Add 'heading' point property to trajectory

    This function will use the coordinates in a trajectory to compute a
    heading for the moving object at each point at the trajectory.
    It will overwrite any heading information already there.

    The input trajectory is modified in place.  The heading at point 0
    is assumed to be the same as at point 1.

    Arguments:
        trajectory (Tracktable terrestrial trajectory): Trajectory
            to modify

    Keyword Arguments:
        property_name (str): Name for the property to add.
            Defaults to "heading".

    Returns:
        Input trajectory (same pointer as was passed in)
    """

    for i in range(1, len(trajectory)):
        last_point = trajectory[i-1]
        this_point = trajectory[i]
        trajectory[i].properties[property_name] = tracktable.core.geomath.bearing(last_point, this_point)
    trajectory[0].properties[property_name] = trajectory[1].properties[property_name]
    return trajectory


def recompute_speed(trajectory: TerrestrialTrajectory,
                     property_name: str="speed") -> TerrestrialTrajectory:
    """Add 'speed' point property to trajectory

    This function will use the coordinates and timestamps in a
    trajectory to compute a speed (in knots) for the moving
    object at each point at the trajectory.  It will overwrite
    any speed information already there.

    The input trajectory is modified in place.  The speed at point
    0 is assumed to be the same as the speed at point 1.

    NOTE: The speed values that come out are fairly sensitive to
    noise in the input data.

    Arguments:
        trajectory (Tracktable terrestrial trajectory): Trajectory
            to modify

    Keyword Arguments:
        property_name (str): Name for the property to add.
            Defaults to "speed".

    Returns:
        Input trajectory (same pointer as was passed in)
    """


    for i in range(1, len(trajectory)):
        last_point = trajectory[i-1]
        this_point = trajectory[i]
        # speed_between() calculates speed in km/h but we want it
        # in knots, hence the multiplier
        trajectory[i].properties[property_name] = tracktable.core.geomath.speed_between(last_point, this_point) * 1.94384
    trajectory[0].properties[property_name] = trajectory[1].properties[property_name]