"""Copyright (c) 2009-2010 Arizona Board of Regents.
All Rights Reserved.
Contact: Lina Karam (karam@asu.edu) and Niranjan Narvekar (nnarveka@asu.edu) 
Image, Video, and Usabilty (IVU) Lab, http://ivulab.asu.edu , Arizona State University

This copyright statement may not be removed from any file containing it or from modifications to these files.
This copyright notice must also be included in any file or product that is derived from the source files.

Redistribution and use of this code in source and binary forms,  with or without modification, are permitted provided that the following conditions are met:
- Redistribution's of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistribution's in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- The Image, Video, and Usability Laboratory (IVU Lab, http://ivulab.asu.edu) is acknowledged in any publication tha reports research results using this code, copies of this code, or modifications of this code.
The code and our papers are to be cited in the bibliography as:

N. D. Narvekar and L. J. Karam, "CPBD Sharpness Metric Software", http://ivulab.asu.edu/Quality/CPBD

N. D. Narvekar and L. J. Karam, "A No-Reference Image Blur Metric Based on the Cumulative
Probability of Blur Detection (CPBD)," accepted and to appear in the IEEE Transactions on Image Processing,  2011.

N. D. Narvekar and L. J. Karam, "An Improved No-Reference Sharpness Metric Based on the Probability of Blur Detection," International Workshop on Video Processing and Quality Metrics for Consumer Electronics (VPQM), January 2010, http://www.vpqm.org (pdf)

N. D. Narvekar and L. J. Karam, "A No Reference Perceptual Quality Metric based on Cumulative Probability of Blur Detection," First International Workshop on the Quality of Multimedia Experience (QoMEX), pp. 87-91, July 2009.

 DISCLAIMER:
 This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the Arizona Board of Regents, Arizona State University, IVU Lab members, authors or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import numpy as np
from scipy.ndimage import convolve
from skimage.filters.edges import HSOBEL_WEIGHTS


def sobel(image: np.ndarray) -> np.ndarray:
    """
    Find edges using the Sobel approximation to the derivatives.

    Inspired by the [Octave implementation](https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l196).
    """

    h1 = np.array(HSOBEL_WEIGHTS)
    h1 /= np.sum(abs(h1))  # normalize h1

    strength2 = np.square(convolve(image, h1.T))

    # Note: https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l59
    thresh2 = 2 * np.sqrt(np.mean(strength2))

    strength2[strength2 <= thresh2] = 0
    return _simple_thinning(strength2)


def _simple_thinning(strength: np.ndarray) -> np.ndarray:
    """
    Perform a very simple thinning.

    Inspired by the [Octave implementation](https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l512).
    """
    num_rows, num_cols = strength.shape

    zero_column = np.zeros((num_rows, 1))
    zero_row = np.zeros((1, num_cols))

    x = (strength > np.c_[zero_column, strength[:, :-1]]) & (
        strength > np.c_[strength[:, 1:], zero_column]
    )

    y = (strength > np.r_[zero_row, strength[:-1, :]]) & (
        strength > np.r_[strength[1:, :], zero_row]
    )

    return x | y
