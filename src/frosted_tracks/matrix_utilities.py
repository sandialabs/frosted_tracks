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

"""Matrix utilities

This contains utilities for converting the inverse covariance matrices
that come from TICC into correlation matrices.

Functions:
    inverse_covariance_to_correlation

"""

import numpy as np
from typing import List, Optional, Union


def inverse_covariance_to_correlation(inverse_covariance_matrices: List[np.ndarray],
                                      window_size: int,
                                      num_sensors: int) -> List[np.ndarray]:
    """Utility function: Convert a list of inverse covariance matrices to a list of correlation matrices.

    You will not need to call this yourself.

    Arguments:
        inverse_covariance_matrices {list of NumPy array}: Matrices to convert
        window_size {int}: Window size used for TICC
        num_sensors {int}: Number of sensors used to create data series

    Returns:
        List of correlation matrices corresponding to inputs
    """

    covariance_matrices = [np.linalg.inv(inv_cov_mat)
                           for inv_cov_mat in inverse_covariance_matrices]
    correlation_matrices = [(np.diag(1.0 / (np.sqrt(np.diag(cov_mat))))
                             @ cov_mat
                             @ np.diag(1.0 / (np.sqrt(np.diag(cov_mat)))))
                            for cov_mat in covariance_matrices]
    reformatted_correlation_matrices = reformat_correlation_matrices(correlation_matrices,
                                                                     window_size,
                                                                     num_sensors)

    return reformatted_correlation_matrices



def reformat_correlation_matrices(thetas: List[np.ndarray],
                                  window_size: int,
                                  num_sensors: int) -> np.ndarray:
    """Given the thetas (ticc_results.markov_random_fields)
    returns a reformated array which matches the Toeplitz matrix laid out
    in the TICC paper

    Arguments:
        thetas (List(np.array(nw, nw))): a List of arrays where each array is a nw x nw
        matrix representing theta
        window_size (int): size of the window
        number_of_sensors (int): number of sensors

    Returns:
        np.array of dimension (k, w, w, n, n); see Toeplitz Matrices under Problem Setup
        in the TICC paper
    """
    n = num_sensors
    w = window_size
    reformated_thetas = np.ndarray(((len(thetas),w,w,n,n)))
    for t in range(len(thetas)):
        for curr_time_offset in range(w):
            for curr_sensor in range(n):
                reformated_thetas[t][curr_time_offset][curr_sensor] = thetas[t][curr_time_offset*n:
                                                                                (curr_time_offset+1)*n,
                                                                                curr_sensor*n:
                                                                                (curr_sensor+1)*n]
    return reformated_thetas

