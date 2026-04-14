# Copyright 2026 National Technology & Engineering Solutions of Sandia,
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

"""Pytest test configuration for Frosted Tracks"""

import functools
import pathlib
import pickle
import pytest

# Additional fixtures can be found in these modules.
pytest_plugins = [
    "tests.fixtures.common",
    "tests.fixtures.features",
    "tests.fixtures.ticc",
    "tests.fixtures.trajectories"
]

# This is the infrastructure that lets us update/retrieve
# golden masters.
#
# A 'golden master' is a piece of output cached on disk that
# a test must replicate in order to succeed.  We use this to
# check for regressions: if anything changes about the content,
# something's gone wrong.
#
# This was inspired by T. Ben Thompson's blog post "How I Test",
# accessed on 14 Apr 2026 at https://tbenthompson.com/post/how_i_test/ 

def pytest_addoption(parser):
    """Add the '--save-golden-masters' command-line option"""
    parser.addoption(
        "--save-golden-masters",
        action="store_true",
        help="Reset golden master files (remember to commit!)"
    )


def _golden_master_directory() -> pathlib.Path:
    test_directory = pathlib.Path(__file__).resolve().parent
    return test_directory / "golden_masters"


def with_golden_master():
    """Decorator to make a test use golden master files

    If you passed the command-line argument '--save-golden-masters'
    to Pytest, this decorator will save the output of your test case
    in the directory '$root/tests/golden_masters/' under the filename
    'test_name.pkl'.  If you did not (i.e. just ran pytest), the
    decorator will load the contents of that file, run the test, 
    and then assert that golden_master == test_output.
    """

    def decorator(run_test):
        try:
            regenerate_masters = pytest.config.getoption("--save-golden-masters")
        except AttributeError as e:
            regenerate_masters = False
            
        @functools.wraps(run_test)
        def wrapper(request, *args, **kwargs):
            test_result = run_test(request, *args, **kwargs)
            
            gm_path = _golden_master_directory()
            if not gm_path.is_dir():
                gm_path.mkdir()

            test_name = request.node.name
            golden_master_filename = gm_path() / f"{test_name}.pkl"

            if regenerate_masters:
                with open(golden_master_filename, "wb") as outfile:
                    pickle.dump(test_result, outfile)
            else:
                with open(golden_master_filename, "rb") as infile:
                    golden_master = pickle.load(infile)
                assert test_result == golden_master
        return wrapper

    return decorator



        
