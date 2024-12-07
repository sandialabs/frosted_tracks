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

"""Find generate_ground_truth functions below a base module"""

from frosted_tracks.test.infrastructure import package_enumeration
from frosted_tracks.test.infrastructure import ground_truth

from types import ModuleType
from typing import Callable, List, Tuple

GroundTruthGenerator = Callable[[ground_truth.GroundTruthOracle], None]


def find_ground_truth_generators(root: ModuleType, fn_name: str="generate_ground_truth") -> List[Tuple[GroundTruthGenerator, ModuleType]]:
    """Find ground truth generators

    Given a module, find functions with a particular name.  Return
    those functions along with the modules containing them.

    The intended use of this function is to collect a list of
    functions to call elsewhere.

    See frosted_tracks.test.infrastructure.package_enumeration for caveats
    on what kinds of modules will and will not be found by this
    search.

    Arguments:
        root (module): Root of module tree to search.  This module
            and all its descendants will be examined for functions
            with a specified name.

    Keyword Arguments:
        fn_name (str): Name of function to search for.  Defaults to
            "generate_ground_truth".

    Returns:
        List of (function, module) tuples.  You can use the module
        to retrieve things like the full name.
    """
    result = []

    # The root package will be at the head of this list
    for child in package_enumeration.all_descendant_packages(root):
        if hasattr(child, fn_name):
            result.append((getattr(child, fn_name), child))
    return result
