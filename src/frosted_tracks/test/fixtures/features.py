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

"""Building blocks for PyTest fixtures for feature functions"""

import tracktable.core.geomath

import frosted_tracks.feature_functions


def _altitude_at_point(p):
    # Apparently boost::python::map_indexing_suite doesn't provide get().
    if "altitude" in p.properties:
        return p.properties["altitude"]
    else:
        return 0


@frosted_tracks.feature_functions.register_feature_function
def altitude(trajectory):
    return [_altitude_at_point(point) for point in trajectory]


@frosted_tracks.feature_functions.register_feature_function
def change_in_altitude(trajectory):
    altitudes = altitude(trajectory)
    result = [altitudes[i] - altitudes[i-1] for i in range(1, len(trajectory))]
    return result + [0]


@frosted_tracks.feature_functions.register_feature_function
def signed_heading_change(trajectory):
    bearing = tracktable.core.geomath.bearing
    headings = [bearing(trajectory[i-1], trajectory[i]) for i in range(1, len(trajectory))]
    headings.append(headings[-1])
    result = []
    for i in range(1, len(headings)):
        dh = headings[i] - headings[i-1]
        if dh > 180:
            dh -= 360
        if dh < -180:
            dh += 360
        result.append(dh)
    result.append(0)
    return result


@frosted_tracks.feature_functions.register_feature_function
def heading(trajectory):
    bearing = tracktable.core.geomath.bearing
    headings = [bearing(trajectory[i-1], trajectory[i]) for i in range(1, len(trajectory))]
    headings.append(headings[-1])
    return headings


@frosted_tracks.feature_functions.register_feature_function
def speed(trajectory):
    speed_lookup = tracktable.core.geomath.speed_between
    result = [speed_lookup(trajectory[i-1], trajectory[i]) for i in range(1, len(trajectory))]
    return result + [result[-1]]


@frosted_tracks.feature_functions.register_feature_function
def change_in_speed(trajectory):
    speeds = speed(trajectory)
    d_speeds = [0] + [speeds[i] - speeds[i-1] for i in range(1, len(trajectory))]
    return d_speeds


def simple_sensors():
    return ["heading", "signed_heading_change", "speed", "change_in_speed"]

