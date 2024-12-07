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


"""Type hints for Frosted Tracks."""

__all__ = [
    "Trajectory",
    "TrajectoryPoint",
    "TerrestrialTrajectory",
    "TerrestrialTrajectoryPoint",
    "Cartesian3DTrajectory",
    "Cartesian3DTrajectoryPoint",
    "FeatureFunction"
]

from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import tracktable.domain.terrestrial
import tracktable.domain.cartesian2d
import tracktable.domain.cartesian3d

# For type hinting
TerrestrialTrajectory = tracktable.domain.terrestrial.Trajectory
Cartesian2DTrajectory = tracktable.domain.cartesian2d.Trajectory
Cartesian3DTrajectory = tracktable.domain.cartesian3d.Trajectory

Trajectory = Union[TerrestrialTrajectory, Cartesian2DTrajectory, Cartesian3DTrajectory]

TerrestrialTrajectoryPoint = tracktable.domain.terrestrial.TrajectoryPoint
Cartesian2DTrajectoryPoint = tracktable.domain.cartesian2d.TrajectoryPoint
Cartesian3DTrajectoryPoint = tracktable.domain.cartesian3d.TrajectoryPoint
TrajectoryPoint = Union[TerrestrialTrajectoryPoint, Cartesian2DTrajectoryPoint, Cartesian3DTrajectoryPoint]

FeatureFunction = Callable[[Trajectory], List[float]]

ClusterableItem = TypeVar('ClusterableItem')
ItemWithId = Tuple[ClusterableItem, int]
DistanceFunction = Callable[[ClusterableItem, ClusterableItem], float]