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

"""Utilities for saving, loading, and discovering ground truth

Most of our test cases use precomputed inputs and outputs to verify
that a piece of code does what we expect it to.  We need three capabilities
in order to make that work:

- Generate ground truth: Find all functions in a package that generate
      ground truth data and run them.
- Save ground truth: Save ground truth files to a known location.
- Load ground truth: Load ground truth files from that known location.

Functions:

generate_ground_truth(root_module)
save_ground_truth(data, key, owning_module)
load_ground_truth(key, owning_module)

"""

import importlib
import logging
import pkgutil

from types import ModuleType
from typing import List

LOGGER = logging.getLogger(__name__)

def _expand_package_name(parent_name: str, subpackage_name: str):
    """Turn a potentially-abbreviated name into a fully qualified name

    Given a parent package name (frosted_tracks.test.foo.bar) and a subpackage name
    (my_subpackage), construct the name frosted_tracks.test.foo.bar.my_subpackage.

    Arguments:
        parent_name (str): Name of package whose children are being enumerated
        subpackage_name (str): Name of child package

    Returns:
        Full name of subpackage

    Raises:
        ValueError: Subpackage name must contain only letters, numbers, and
        underscores.
    """

    if not subpackage_name.isidentifier():
        raise ValueError((
            f"Subpackage name {subpackage_name} must only contain letters, "
            f"numbers, and underscores."
        ))

    return f"{parent_name}.{subpackage_name}"


def child_package_names(root_package: ModuleType):
    """List all immediate children of a package

    If a package has any immediate children discoverable by pkgutil, this
    function will list their names.  If it does not, it will return an
    empty list.

    NOTE: This does not work with modules that play games with dynamic
          imports.  For example, the built-in 'os' module decides what
          platform-specific module to load as 'os.path' when it is first
          imported.  On POSIX systems, that module is 'posixpath'.  We
          will find posixpath under its proper name but not find it
          under its alias os.path.

    Example:
        >>> import xml
        >>> _child_package_names(xml)
        ['dom', 'etree', 'parsers', 'sax']

    Arguments:
        root_package (module): Package whose children you want

    Returns:
        List of strings with child package names.  Note that these names
        are just the last component of the name: you will need to combine
        them with the names of the parents if you want the full name.
    """

    if not hasattr(root_package, "__path__"):
        return []
    else:
        result = [module_info.name
                  for module_info in pkgutil.iter_modules(root_package.__path__)]
        return result


def all_descendant_packages(root: ModuleType) -> List[ModuleType]:
    """Return a list of all subpackages underneath a root

    This will walk the package tree (pre-order traversal), import
    each child, and return all the packages in the list in which
    they were visited.

    NOTE: The root package will be the first element in the result
          list.

    Arguments:
        root (module): Module whose descendants you want

    Returns:
        List of imported module objects
    """

    all_modules = [root]

    if hasattr(root, "__path__"):
        child_names = child_package_names(root)
        for child_name in child_names:
            full_name = f"{root.__name__}.{child_name}"
            try:
                child_package = importlib.import_module(full_name)
                all_modules.extend(all_descendant_packages(child_package))
            except ModuleNotFoundError as exc:
                LOGGER.warning((
                    f"Non-fatal error while importing module {full_name}: "
                    f"{exc.msg}"
                ))
                continue
            except ImportError as exc:
                LOGGER.warning((
                    f"Non-fatal error while importing {full_name}: "
                    f"{exc.msg}"
                ))
                continue

    return all_modules

