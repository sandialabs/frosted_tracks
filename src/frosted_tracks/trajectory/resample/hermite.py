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

"""Internal methods for Hermite resampling of coordinates"""

__all__ = ["resample_by_time_hermite"]

import datetime
import math
from typing import Callable, Optional

import numpy as np
import pyproj
import scipy.interpolate

import tracktable.domain.terrestrial
import tracktable.domain.cartesian3d

from frosted_tracks.frosted_tracks_types import (
    Trajectory, TrajectoryPoint,
    TerrestrialTrajectory, TerrestrialTrajectoryPoint,
    Cartesian3DTrajectory, Cartesian3DTrajectoryPoint
)

def _check_property_valid(trajectory: Trajectory, property: str) -> None:
    """Check to see if a certain property is present and non-null

    This function will check a trajectory to make sure a named
    property is present and non-null (None in Python)
    at all points.

    Arguments:
        trajectory (Tracktable trajectory): Trajectory to check
        property_name (str): Property to check

    Raises:
        ValueError: Some trajectory point is missing the property
            or its value is None

    No return value.  If the function returns without throwing an
    exception, all is well.
    """

    for point in trajectory:
        if property not in point.properties:
            raise ValueError((
                f"_check_property_valid: One or more points in the source "
                f"trajectory is missing the required property '{property}'.  "
                f"Please ensure that this value is present and valid on all "
                f"points before using Hermite interpolation."
            ))

    for point in trajectory:
        if point.properties[property] is None:
            raise ValueError((
                f"_check_property_valid: The property '{property}' is "
                f"present on all points in the source trajectory but its "
                f"value is None in at least one case.  Please ensure that "
                f"this property is present and valid on all points before "
                f"using Hermite interpolation."
            ))



def _copy_generic_properties(src, dest):
    for (name, value) in src.properties.items():
        dest.properties[name] = value


def _convert_trajectory(trajectory: Trajectory,
                        new_trajectory_type,
                        point_converter: Callable,
                        converter_kwargs: dict):
    """Internal method - convert a trajectory one point at a time

    Given a function that converts a trajectory point, this will
    instantiate a new trajectory, convert all the points, and add
    them into the new trajectory.

    Arguments:
        trajectory (Tracktable trajectory): Trajectory to convert
        new_trajectory_type (class): type (NOT an instance) to convert
            to
        point_converter (callable): Function from input trajectory's
            point type to output trajectory's point type
        converter_kwargs (dict): Any extra arguments you want to
            pass to the converter

    Returns:
        Trajectory of specified new type
    """

    result = new_trajectory_type()
    _copy_generic_properties(trajectory, result)

    first_point = True
    for src_point in trajectory:
        new_point = point_converter(src_point, **converter_kwargs)
        if first_point:
            first_point = False
        result.append(new_point)

    return result


# Possible optimization: create the transformer once and then
# just pass it in

def _geodetic_to_geocentric(longitude, latitude, altitude):
    """Convert LLA (longitude/latitude/altitude) to geocentric coordinates

    Arguments:
        longitude {float}: Decimal degrees of longitude
        latitude {float}: Decimal degrees of latitude
        altitude {float}: Meters (?) of altitude above the WGS84 datum

    Note: Geocentric coordinates are also known as ECEF (Earth Centered /
    Earth Fixed).
    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"}
    )
    return transformer.transform(longitude, latitude, altitude)


def _geocentric_to_geodetic(x, y, z):
    """Convert geocentric coordinates to LLA (longitude/latitude/altitude)

    Note: Geocentric coordinates are also known as ECEF (Earth Centered/
    Earth Fixed) coordinates.

    Arguments:
        x {float}: X coordinate in geocentric frame
        y {float}: Y coordinate in geocentric frame
        z {float}: Z coordinate in geocentric frame
    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"}
    )
    return transformer.transform(x, y, z, radians=False)


def _copy_point_properties(src_point, new_point):
    """Copy all properties from one TraejctoryPoint to another"""
    new_point.object_id = src_point.object_id
    new_point.timestamp = src_point.timestamp
    _copy_generic_properties(src_point, new_point)
    return new_point

def _trajectory_point_geodetic_to_geocentric(geodetic_point: TerrestrialTrajectoryPoint,
                                             altitude_name: Optional[str] = "altitude",
                                             altitude_units: str="feet") -> Cartesian3DTrajectoryPoint:
    """Convert a trajectory point from geodetic coordinates to geocentric

    Geodetic coordinates are the familiar longitude/latitude.

    Geocentric coordinates are a 3D Cartesian coordinate frame where
    points are measured as distances in meters from the mass center
    of the Earth.

    Properties are left unchanged.  Altitude is calculated as described
    below if "altitude_name" is supplied; otherwise left at 0 (mean sea level).

    Arguments:
        geodetic_point (terrestrial trajectory point): Point to convert

    Keyword Arguments:
        altitude_name (str): If provided, this is used as the name
            of a property that contains the point's altitude.
            Defaults to "altitude".  If None, altitude is set to
            zero.
        altitude_units (str): "feet" or "meters"

    Returns:
        New trajectory point in Tracktable Cartesian 3D domain with geocentric
    """

    result_point = tracktable.domain.cartesian3d.TrajectoryPoint()
    _copy_point_properties(geodetic_point, result_point)

    altitude = 0
    if (altitude_name is not None and altitude_name != ""):
        altitude = geodetic_point.properties["altitude"]
        if altitude_units == "feet":
            altitude *= 12 / 39.37 # convert to meters
        elif altitude_units != "meters":
            raise ValueError("Altitude units must be either 'feet' or 'meters'.")

    (x, y, z) = _geodetic_to_geocentric(geodetic_point[0], geodetic_point[1], altitude)

    result_point[0] = x
    result_point[1] = y
    result_point[2] = z

    return result_point

def _trajectory_geodetic_to_geocentric(geodetic_trajectory: TerrestrialTrajectory,
                                      altitude_name: Optional[str] = "altitude",
                                      altitude_units: str="feet") -> Cartesian3DTrajectory:
    """Convert a trajectory from geodetic coordinates to geocentric

    Geodetic coordinates are the familiar longitude/latitude.

    Geocentric coordinates are a 3D Cartesian coordinate frame where
    points are measured as distances in meters from the mass center
    of the Earth.

    Properties on point and trajectory are left unchanged.  Altitude
    is calculated as described below if "altitude_name" is specified;
    otherwise left at 0 (mean sea level).

    Arguments:
        geodetic_trajectory (terrestrial trajectory): Trajectory to convert

    Keyword Arguments:
        altitude_name (str): If provided, this is used as the name
            of a property that contains the point's altitude.
            Defaults to "altitude".  If None, altitude is set to
            zero.
        altitude_units (str): "feet" or "meters"

    Returns:
        New trajectory in Tracktable Cartesian 3D domain with geocentric coordinates
    """

    return _convert_trajectory(
        geodetic_trajectory,
        tracktable.domain.cartesian3d.Trajectory,
        _trajectory_point_geodetic_to_geocentric,
        {"altitude_name": altitude_name, "altitude_units": altitude_units}
    )


def _trajectory_point_geocentric_to_geodetic(geocentric_point: Cartesian3DTrajectoryPoint,
                                             altitude_name: Optional[str] = "altitude",
                                             altitude_units: str="feet") -> TerrestrialTrajectoryPoint:

    """Convert a trajectory point from geocentric coordinates to geodetic

    Geocentric coordinates are a 3D Cartesian coordinate frame where
    points are measured as distances in meters from the mass center
    of the Earth.  Geodetic coordinates are the familiar latitude/longitude.

    Properties are left unchanged.  If "altitude_name" is specified,
    altitude will be calculated and set as described below.

    Arguments:
        geocentric_point (terrestrial trajectory point): Point to convert

    Keyword Arguments:
        altitude_name (str): If provided, this is used as the name
            of a property that will contain the point's calculated altitude.
            Defaults to "altitude".  If None, altitude is not set.add
        altitude_units (str): "feet" or "meters"

    Returns:
        New trajectory point in Tracktable Cartesian 3D domain with geocentric
    """

    result_point = tracktable.domain.terrestrial.TrajectoryPoint()
    _copy_point_properties(geocentric_point, result_point)

    (longitude, latitude, altitude) = _geocentric_to_geodetic(
        geocentric_point[0], geocentric_point[1], geocentric_point[2]
    )

    result_point[0] = longitude
    result_point[1] = latitude

    if (altitude_name is not None and altitude_name != ""):
        if altitude_units == "feet":
            altitude *= 39.37 / 12
        elif altitude_units != "meters":
            raise ValueError("Altitude units must be either 'feet' or 'meters'.")
        result_point.properties[altitude_name] = altitude

    assert type(result_point) == tracktable.domain.terrestrial.TrajectoryPoint
    return result_point


def _trajectory_geocentric_to_geodetic(geocentric_trajectory: Cartesian3DTrajectory,
                                       altitude_name: Optional[str] = "altitude",
                                       altitude_units: str="feet") -> TerrestrialTrajectory:
    """Convert a trajectory from geocentric coordinates to geodetic

    Geocentric coordinates are a 3D Cartesian coordinate frame where
    points are measured as distances in meters from the mass center
    of the Earth.

    Geodetic coordinates are the familiar longitude/latitude/altitude.

    Properties are left unchanged.  If "altitude_name" is specified,
    altitude will be calculated and set as described below.

    Arguments:
        geodetic_trajectory (terrestrial trajectory): Trajectory to convert

    Keyword Arguments:
        altitude_name (str): If provided, this is used as the name
            of a property that contains the point's altitude.
            Defaults to "altitude".  If None, altitude is set to
            zero.
        altitude_units (str): "feet" or "meters"

    Returns:
        New trajectory in Tracktable terrestrial domain with geodetic coordinates
    """

    return _convert_trajectory(
        geocentric_trajectory,
        tracktable.domain.terrestrial.Trajectory,
        _trajectory_point_geocentric_to_geodetic,
        {"altitude_name": altitude_name, "altitude_units": altitude_units}
    )


# These are utility functions for computing velocity vectors in geocentric coordinates

def _north_vector(longitude: float, latitude: float) -> np.ndarray:
    """Compute the geocentric north unit vector for a given longitude and latitude"""

    def d2r(degrees):
        return math.pi * degrees / 180

    longitude = d2r(longitude)
    latitude = d2r(latitude)

    result = [
        - math.cos(longitude) * math.sin(latitude),
        - math.sin(longitude) * math.sin(latitude),
        math.cos(latitude)
    ]
    return np.array(result)


def _east_vector(longitude: float, latitude: float) -> np.ndarray:
    """Compute the geocentric east unit vector for a given longitude and latitude"""

    def d2r(degrees):
        return math.pi * degrees / 180

    longitude = d2r(longitude)
    latitude = d2r(latitude)
    result = [
        - math.sin(longitude), math.cos(longitude), 0
    ]
    return np.array(result)

def _knots_to_m_per_sec(knots):
    """Compute speed in knots to meters per second

    """
    return knots * 0.514444

def _geocentric_velocity_at_point(point: TerrestrialTrajectoryPoint,
                                  speed_name: str="speed",
                                  heading_name: str="heading",
                                  speed_conversion_factor: float=0.514444) -> np.ndarray:
    """Compute velocity vector at a point in geocentric coordinates

    Given position in longitude/latitude, speed, and heading,
    construct a 3D velocity vector whose direction is the object's
    velocity in geocentric coordinates and whose length is the
    object's speed in meters per second.

    Arguments:
        point (Tracktable terrestrial point): Trajectory with
            geodetic (longitude/latitude) coordinates

    Keyword Arguments:
        speed_name (string): Name of attribute containing speed.  Defaults
            to "speed".
        heading_name (string): Name of attribute containing heading/bearing.
            Defaults to "heading".
        speed_conversion_factor (float): Conversion factor to translate speeds
            as stored on points into speeds in meters per second.  Defaults to
            0.514444, which is one knot measured in meters per second.

    Returns:
        3 x 1 NumPy array containing X, Y, and Z components of velocity vector
    """

    def d2r(degrees):
        return math.pi * degrees / 180.0

    north = _north_vector(point[0], point[1])
    east = _east_vector(point[0], point[1])
    heading = point.properties[heading_name]
    speed_in_ms = speed_conversion_factor * point.properties[speed_name]

    #print(f"At ({point[0]}, {point[1]}), north vector is {north} and east_vector is {east}")
    unit_velocity = north * math.cos(d2r(heading)) + east * math.sin(d2r(heading))
    scaled_velocity = speed_in_ms * unit_velocity
    return scaled_velocity




def _geocentric_velocity_vectors(trajectory: TerrestrialTrajectory,
                                 speed_name: str="speed",
                                 heading_name: str="heading",
                                 speed_conversion_factor: float=0.514444) -> np.ndarray:
    """Compute velocity vectors in geocentric coordinates

    Given position, speed, and heading, construct a 3D velocity
    vector whose direction is the object's velocity in geocentric
    coordinates and whose length is the object's speed in meters
    per second.

    Arguments:
        trajectory (Tracktable terrestrial trajectory): Trajectory with
            geodetic (longitude/latitude) coordinates

    Keyword Arguments:
        speed_name (string): Name of attribute containing speed.  Defaults
            to "speed".
        heading_name (string): Name of attribute containing heading/bearing.
            Defaults to "heading".
        speed_conversion_factor (float): Conversion factor to translate speeds
            as stored on points into speeds in meters per second.  Defaults to
            0.514444, which is one knot measured in meters per second.

    Returns:
        3 x N NumPy array, one column for each point.  Rows are X, Y, and Z
        components of velocity vectors.
    """

    result = np.zeros(shape=(3, len(trajectory)))

    for (i, point) in enumerate(trajectory):
        result[:, i] = _geocentric_velocity_at_point(
                            point,
                            speed_name=speed_name, heading_name=heading_name,
                            speed_conversion_factor=speed_conversion_factor)

    return result


def _resample_coordinate(x: np.ndarray,
                         y: np.ndarray,
                         dydx: np.ndarray,
                         new_x_values: np.ndarray) -> np.ndarray:
    """Resample a single coordinate y = f(x) with a cubic Hermite spline

    Arguments:
        x (array-like): X values for points to fit
        y (array-like): Y values for points to fit
        dydx (array-like): Derivatives of function to approximate at points to fit
        new_x_values (array-like): Values of X for which to interpolate values for Y

    Returns:
        Array of N values containing f(x) for each new X value
    """

    interpolator = scipy.interpolate.CubicHermiteSpline(x, y, dydx)
    result = []
    for new_x in new_x_values:
        result.append(interpolator(new_x))
    return np.array(result)


def resample_by_time_hermite(trajectory_to_fit: TerrestrialTrajectory,
                             interval: datetime.timedelta,
                             altitude_name: Optional[str]=None,
                             speed_name: str="speed",
                             heading_name: str="heading",
                             speed_conversion_factor: float=0.514444,
                             include_last_point: bool=False) -> TerrestrialTrajectory:
    """Resample a trajectory by fitting an Hermite spline

    The difference between this and the PCHIP interpolator is that
    the Hermite spline matches derivatives.  When we have velocity
    information, that should let us do a better job of rebuilding
    the trajectory.

    Arguments:
        trajectory_to_fit (Tracktable terrestrial trajectory): Source trajectory to
            resample
        interval (datetime.timedelta): Spacing between points in
            resampled trajectory

    Keyword Arguments:
        altitude_name (string): Name of point property containing altitude.  Defaults
            to None (no altitude available).
        speed_name (string): Name of point property containing speed.  Defaults to
            "speed".
        heading_name (string): Name of point property containing heading.  Defaults
            to "heading".
        speed_conversion_factor (float): Factor relating speed units to meters per
            second.  Defaults to 0.514444: one knot is 0.514444 meters per second.
        include_last_point (bool): If true, the last point in the original trajectory
            will be included whether or not it falls on the desired spacing.
            Defaults to False.

    """

    _check_property_valid(trajectory_to_fit, speed_name)
    _check_property_valid(trajectory_to_fit, heading_name)

    start_time = trajectory_to_fit[0].timestamp
    end_time = trajectory_to_fit[-1].timestamp
    duration = end_time - start_time

    # These are the X values for the data series we'll fit
    elapsed_seconds = [(p.timestamp - start_time).total_seconds() for p in trajectory_to_fit]

    new_point_count = int(math.floor(duration / interval))
    new_t_values = interval.total_seconds() * np.arange(new_point_count)
    new_timestamps = [start_time + i * interval for i in range(new_point_count)]

    if include_last_point:
        if end_time < new_timestamps[-1]:
            new_point_count += 1
            new_t_values.append(1)
            new_timestamps.append(end_time)

    # We'll use these to get the derivatives in geocentric coordinates
    dxdydz_dt = _geocentric_velocity_vectors(trajectory_to_fit,
                                             speed_name=speed_name,
                                             heading_name=heading_name,
                                             speed_conversion_factor=speed_conversion_factor)
    geocentric = _trajectory_geodetic_to_geocentric(trajectory_to_fit, altitude_name=altitude_name)

    # Resample each coordinate separately
    gc_x = [point[0] for point in geocentric]
    gc_y = [point[1] for point in geocentric]
    gc_z = [point[2] for point in geocentric]
    dx_dt = dxdydz_dt[0, :]
    dy_dt = dxdydz_dt[1, :]
    dz_dt = dxdydz_dt[2, :]

    new_x = _resample_coordinate(elapsed_seconds, gc_x, dx_dt, new_t_values)
    new_y = _resample_coordinate(elapsed_seconds, gc_y, dy_dt, new_t_values)
    new_z = _resample_coordinate(elapsed_seconds, gc_z, dz_dt, new_t_values)

    resampled_geocentric = tracktable.domain.cartesian3d.Trajectory()
    # Don't bother copying the trajectory properties -- all we really want
    # are the coordinates

    for (i, timestamp) in enumerate(new_timestamps):
        # Start by interpolating all the properties
        resampled_geocentric.append(tracktable.core.geomath.point_at_time(geocentric, timestamp))
        # Replace the coordinates with what we got from the spline
        resampled_geocentric[i][0] = new_x[i]
        resampled_geocentric[i][1] = new_y[i]
        resampled_geocentric[i][2] = new_z[i]

    # Convert back to geodetic to get the coordinates for the new points.  Trust the
    # altitude values from the original.
    resampled_geodetic = _trajectory_geocentric_to_geodetic(resampled_geocentric,
                                                            altitude_name=None)

    return resampled_geodetic
