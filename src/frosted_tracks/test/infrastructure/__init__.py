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

"""Support structures for frosted_tracks test cases.

The most important thing here is the function run_ground_truth_generators().

"""
import importlib
import logging

from frosted_tracks.test.infrastructure import find_generators
from frosted_tracks.test.infrastructure import ground_truth

LOGGER = logging.getLogger(__name__)

def run_ground_truth_generators():
    """Find and run all ground truth generators in frosted_tracks.test

    When you have a test case that needs to use known-good artifacts on disk
    as inputs or to compare against known-good outputs, write a function in
    your test module named generate_ground_truth(oracle).  The oracle argument
    should be a GroundTruthOracle as defined in
    frosted_tracks.test.infrastructure.ground_truth.  Your generate_ground_truth()
    function should build and store everything you need.

    As a test writer, be aware that the results of floating-point calculations
    may not be the same between platforms, especially when trigonometry
    and logarithms are involved.  Storing inputs on disk (so you know exactly
    what the input bits are) and testing for approximate equality on output
    are both good ways to mitigate this.

    When invoked, run_ground_truth_generators() will walk through all of
    frosted_tracks.test, collect those functions, and call them with appropriately
    configured oracles.

    No arguments.  Returns None.
    """

    frosted_tracks_test = importlib.import_module("frosted_tracks.test")
    for (gt_generator, module) in find_generators.find_ground_truth_generators(frosted_tracks_test):
        oracle = ground_truth.GroundTruthOracle(module.__name__)
        LOGGER.info("Generating ground truth for %s", module.__name__)
        gt_generator(oracle)


