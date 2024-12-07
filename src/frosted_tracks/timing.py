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

"""Decorator and context manager to time code execution"""

# This is from https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator

import functools
import math
import time

def timing(fn):
    """Decorator to time function execution

    Use this like any other decorator:

    .. code-block: python

    @timing
    def my_function(*args, **kwargs):
        # Do something
        return None


    Each time you execute ``my_function`` thereafter, a message will
    be printed after it returns:

    Function my_function([1, 2, 3], {'my_keyword': 456}) took 3.5 seconds

    """


def timing(fn):
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        end_time = time.perf_counter_ns()
        duration_ns = end_time - start_time

        duration_str = _readable_duration(duration_ns)
        print((
            f"Function {fn.__name__}({args}, {kwargs}) took {duration_str} seconds"
        ))
        return result
    return wrap


def _readable_duration(duration_ns: int) -> str:
    """Convert a time measured in nanoseconds to readable form

    If time is less than 1 microsecond, display with nanoseconds
    as units.  If time is less than 1 millisecond, display with
    microseconds as units.  Keep going like this.

    Arguments:
        duration_ns (int): Duration to print, measured in nanoseconds

    Returns:
        String representation of duration
    """

    one_microsecond = 1000
    one_millisecond = 1000000
    one_second = 1000000000
    one_minute = 60 * one_second
    one_hour = 60 * one_minute

    if duration_ns < one_microsecond:
        suffix = "ns"
        scaled_duration = duration_ns
    elif duration_ns < one_millisecond:
        suffix = "μs"
        scaled_duration = duration_ns / one_microsecond
    elif duration_ns < one_second:
        suffix = "ms"
        scaled_duration = duration_ns / one_millisecond
    elif duration_ns < one_minute:
        suffix = "seconds"
        scaled_duration = duration_ns / one_second
    else:
        suffix = "minutes"
        scaled_duration = duration_ns / one_minute

    if suffix == "ns":
        return f"{scaled_duration} {suffix}"
    else:
        return f"{scaled_duration:.4} {suffix}"