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

"""Ground truth oracle

Most of our test cases use precomputed inputs and outputs to verify
that a piece of code does what we expect it to.  We need three capabilities
in order to make that work:

1. Generate ground truth: Find all functions in a package that generate
      ground truth data and run them.
2. Save ground truth: Save ground truth files to a known location.
3. Load ground truth: Load ground truth files from that known location.

This module contains the ground truth oracle, a class that handles #2 and
#3 on that list.  #1 is in frosted_tracks.test.infrastructure.

"""

import os.path
import pickle
from typing import Any, Optional

def here():
    """Directory containing this file.

    This should be ...repository/src/frosted_tracks/test/infrastructure.
    """
    return os.path.dirname(os.path.realpath(__file__))


def test_data_directory():
    return os.path.normpath(os.path.join(here(), "..", "test_data"))


class GroundTruthOracle:
    """Repository for ground truth data

    This class manages storage of data artifacts used to provide test input
    and compare output with known good results.

    By default, artifacts will be stored in a common directory for the
    package as a whole.  We recommend supplying the 'owning_module_name'
    keyword argument to organize this.

    Data artifacts are saved and restored using Python's pickle module.

    Note that the owning module name and the keys for artifacts can only
    contain letters, numbers, spaces, dashes, and underscores.


    Methods:
        put(data, name): Save a data artifact
        get(name): Load a previously-saved artifact
        exists(name): Check to see whether an artifact exists
    """

    def __init__(self, owning_module_name: Optional[str]=None):
        if owning_module_name is not None:
            _validate_filename(owning_module_name)
            self._data_path = os.path.join(test_data_directory(),
                                           owning_module_name)
        else:
            self._data_path = test_data_directory()


    def put(self, data: Any, key: str, overwrite: bool=False) -> None:
        """Save a data artifact

        Write a new block of data to the archive.  The data can be any size
        and will be saved with pickle.

        The key must contain only letters, numbers, dashes, underscores,
        and spaces.

        Arguments:
            data (any): Thing to save
            key (str): Name under which to save data

        Keyword Arguments:
            overwrite (bool): If True, any existing data will be overwritten.
                If False, attempting to overwrite an existing artifact will
                raise an error.  Defaults to False.

        Raises:
            ValueError: Key contains illegal characters.
            KeyError: Artifact with this key already exists.

        Returns:
            None.
        """

        _validate_filename(key)
        filename = self._make_full_filename(key)
        if os.path.exists(filename) and not overwrite:
            raise KeyError((
                f"A data artifact with key {key} already exists."
            ))
        if not os.path.isdir(self._data_path):
            os.makedirs(self._data_path, exist_ok=True)
        with open(filename, "wb") as outfile:
            pickle.dump(data, outfile)


    def get(self, key: str) -> Any:
        """Load a data artifact

        Retrieve a previously saved artifact.  You must know its key.

        The key must contain only letters, numbers, dashes, underscores,
        and spaces.

        Arguments:
            key (str): Name for artifact to load

        Raises:
            ValueError: Key contains illegal characters.
            KeyError: No artifact with this name exists.

        Returns:
            Data from archive.
        """

        _validate_filename(key)
        filename = self._make_full_filename(key)
        if not os.path.exists(filename):
            raise KeyError((
                f"No data artifact with key {key} exists."
            ))
        with open(filename, "rb") as infile:
            return pickle.load(infile)


    def exists(self, key: str) -> bool:
        """Check to see whether an artifact exists

        Don't try to load the artifact, just check to see if it's there.

        The key must contain only letters, numbers, dashes, underscores,
        and spaces.

        Arguments:
            key (str): Name of artifact to check for

        Raises:
            ValueError: Key contains illegal characters.

        Returns:
            True if artifact is found, false if not.
        """

        _validate_filename(key)
        filename = self._make_full_filename(key)
        return os.path.exists(filename)


    def delete(self, key: str):
        """Delete an existing artifact

        You must know the artifact's key.  It is an error to try to delete
        an artifact that's not actually there.

        The key must contain only letters, numbers, dashes, underscores,
        and spaces.

        Arguments:
            key (str): Name of artifact to check for

        Raises:
            ValueError: Key contains illegal characters.
            KeyError: No artifact with that name was found.

        Returns:
            True if artifact is found, false if not.
        """

        _validate_filename(key)
        filename = self._make_full_filename(key)
        if not os.path.exists(filename):
            raise KeyError((
                f"No data artifact with key {key} exists."
            ))
        return os.remove(filename)

    def _make_full_filename(self, key):
        most_of_path = os.path.join(self._data_path, key)
        if most_of_path.endswith(".pkl"):
            return most_of_path
        else:
            return f"{most_of_path}.pkl"


def _validate_filename(name: str):
    """Catch illegal file names

    We define a legal file name as one that contains only
    alphanumeric characters (as determined by isalpha() and isdigit()),
    dots, and underscores.

    Arguments:
        name (str): Filename to check

    Raises:
        ValueError: contains invalid characters

    Returns:
        No return value.  If the function returns without raising an
        exception, everything's fine.
    """

    for char in name:
        if not (
            char.isalpha()
            or char.isdigit()
            or char == "."
            or char == "_"
        ):
            raise ValueError((
                f"Key or filename f{name} may only contain letters, numbers, '.', and '_'"
            ))
