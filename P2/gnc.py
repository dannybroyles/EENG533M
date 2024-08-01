"""
Copyright 2021 Dr. David Woodburn

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

__author__ = "David Woodburn"
__credits__ = ["John Raquet", "Mike Veth", "Alex McNeil"]
__license__ = "MIT"
__date__ = "2022-05-19"
__maintainer__ = "David Woodburn"
__email__ = "david.woodburn@icloud.com"
__status__ = "Development"

import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import leastsq

# ---------
# Constants
# ---------

# mathematical constants
RAD_TO_DEG = 57.295779513082323     # radians-to-degrees scaling factor
DEG_TO_RAD = 0.017453292519943295   # degrees-to-radians scaling factor

# WGS84 constants (IS-GPS-200M and NIMA TR8350.2)
F_L1_GPS = 1575.42e6        # L1 carrier frequency for GPS [Hz] (p. 14)
F_L2_GPS = 1227.6e6         # L2 carrier frequency for GPS [Hz] (p. 14)
PI_GPS = 3.1415926535898    # pi as defined for GPS calculations (p. 109)
MU_E = 3.986005e14          # Earth's grav. constant [m^3/s^2] (p. 97)
c = 299792458.0             # speed of light [m/s] (p. 98)
W_IE = 7.2921151467e-5      # sidereal Earth rate [rad/s] (p. 106)
A_E = 6378137.0             # Earth's semi-major axis [m] (p. 109)
F_E = 298.257223563         # Earth's flattening constant (NIMA)
B_E = 6356752.314245        # Earth's semi-minor axis [m] A_E*(1 - 1/F_E)
E2 = 6.694379990141317e-3   # Earth's eccentricity squared [ND] (derived)

# -------------------
# Inertial Navigation
# -------------------

def is_square(C, N=3):
    """
    Check if the variable C is a square matrix with length N.

    Parameters
    ----------
    C : (N, N) np.ndarray
        Variable which should be a matrix.
    N : int, default 3
        Intended length of C matrix.

    Returns
    -------
    True if C is a square matrix of length N, False otherwise.
    """

    # Check the inputs.
    if not isinstance(N, int):
        raise Exception('is_square: N must be an integer.')

    if isinstance(C, np.ndarray) and (C.ndim == 2) and (C.shape == (N, N)):
        return True
    else:
        return False


def is_ortho(C):
    """
    Check if the matrix C is orthogonal.  It is assumed that C is a square
    2D np.ndarray (see is_square).

    Parameters
    ----------
    C : (N, N) np.ndarray
        Square matrix.

    Returns
    -------
    True if C is an orthogonal matrix, False otherwise.
    """

    N = len(C)
    Z = np.abs(C @ C.T - np.eye(N))
    check = np.sum(Z.flatten())
    return (check < N*N*1e-15)


def rpy_to_dcm(r, p, y):
    """
    Convert roll, pitch, and yaw angles to a direction cosine matrix that
    represents a zyx sequence of right-handed rotations.

    Parameters
    ----------
    r : float or int
        Roll in radians from -pi to pi.
    p : float or int
        Pitch in radians from -pi/2 to pi/2.
    y : float or int
        Yaw in radians from -pi to pi.

    Returns
    -------
    R : (3, 3) np.ndarray
        Rotation matrix.

    See Also
    --------
    dcm_to_rpy
    rot

    Notes
    -----
    This is equivalent to generating a rotation matrix for the rotation from the
    navigation frame to the body frame.  However, if you want to rotate from the
    body frame to the navigation frame (a xyz sequence of right-handed
    rotations), transpose the result of this function.  This is a convenience
    function.  You could instead use the `rot` function as follows::

        R = rot([yaw, pitch, roll], [2, 1, 0])

    However, the `rpy_to_dcm` function will compute faster than the `rot`
    function.
    """

    # Check inputs.
    if not isinstance(r, (float, int)) or not isinstance(p, (float, int)) or not isinstance(y, (float, int)):
        raise Exception('rpy_to_dcm: r, p, and y must be floats or ints')
    if (abs(r) > np.pi) or (abs(p) > np.pi/2) or (abs(y) > np.pi):
        raise Exception('rpy_to_dcm: r and y must be bound by -pi and pi and ' +
                'p must be bound by -pi/2 and pi/2.')

    # Get the cosine and sine functions of the roll, pitch, and yaw.
    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)
    cy = np.cos(y)
    sy = np.sin(y)

    # Build and return the 3x3 matrix.
    R = np.array([
        [            cp*cy,             cp*sy,   -sp],
        [-cr*sy + sr*sp*cy,  cr*cy + sr*sp*sy, sr*cp],
        [ sr*sy + cr*sp*cy, -sr*cy + cr*sp*sy, cr*cp]])

    return R


def dcm_to_rpy(dcm):
    """
    Convert the direction cosine matrix, `dcm`, to vectors of `roll`, `pitch`,
    and `yaw` (in that order) Euler angles.

    This `dcm` represents the z-y-x sequence of right-handed rotations.  For
    example, if the DCM converted vectors from the navigation frame to the body
    frame, the roll, pitch, and yaw Euler angles would be the consecutive angles
    by which the vector would be rotated from the navigation frame to the body
    frame.  This is as opposed to the Euler angles required to rotate the vector
    from the body frame back to the navigation frame.

    Parameters
    ----------
    dcm : (3, 3) np.ndarray
        Rotation direction cosine matrix.

    Returns
    -------
    r : float
        Intrinsic rotation about the final reference frame's x axis.
    p : float
        Intrinsic rotation about the intermediate reference frame's y axis.
    y : float
        Intrinsic rotation about the initial reference frame's z axis.

    See Also
    --------
    rpy_to_dcm
    rot

    Notes
    -----
    If we define `dcm` as ::

              .-             -.
              |  d11 d12 d13  |
        dcm = |  d21 d22 d23  |
              |  d31 d32 d33  |
              '-             -'
              .-                                                 -.
              |       (cy cp)             (sy cp)          -sp    |
            = |  (cy sp sr - sy cr)  (sy sp sr + cy cr)  (cp sr)  |
              |  (sy sr + cy sp sr)  (sy sp cr - cy sr)  (cp cr)  |
              '-                                                 -'

    where `c` and `s` mean cosine and sine, respectively, and `r`, `p`, and `y`
    mean roll, pitch, and yaw, respectively, then we can see that ::

                                    .-       -.
                                    |  cp sr  |
        r = atan2(d23, d33) => atan | ------- |
                                    |  cp cr  |
                                    '-       -'
                                    .-       -.
                                    |  sy cp  |
        y = atan2(d12, d11) => atan | ------- |
                                    |  cy cp  |
                                    '-       -'

    where the cp values cancel in both cases.  The value for pitch could be
    found from d13 alone::

        p = asin(-d13)

    However, this tends to suffer from numerical error around +/- pi/2.  So,
    instead, we will use the fact that ::

          2     2               2     2
        cy  + sy  = 1   and   cr  + sr  = 1 .

    Therefore, we can use the fact that ::

           .------------------------
          /   2      2      2      2     .--
         V d11  + d12  + d23  + d33  =  V 2  cos( |p| )

    to solve for pitch.  We can use the negative of the sign of d13 to give the
    proper sign to pitch.  The advantage is that in using more values from the
    dcm matrix, we can can get a value which is more accurate.  This works well
    until we get close to a pitch value of zero.  Then, the simple formula for
    pitch is actually better.  So, we will use both and do a weighted average of
    the two, based on pitch.

    References
    ----------
    .. [1]  Titterton & Weston, "Strapdown Inertial Navigation Technology"
    """

    # Check inputs.
    if not is_square(dcm, 3):
        raise Exception('dcm_to_rpy: dcm must be a (3, 3) np.ndarray')
    if not is_ortho(dcm):
        raise Exception('dcm_to_rpy: dcm must be orthogonal')

    # Get roll and yaw.
    r = np.arctan2(dcm[1, 2], dcm[2, 2])
    y = np.arctan2(dcm[0, 1], dcm[0, 0])

    # Get pitch.
    sp = -dcm[0, 2]
    pa = np.arcsin(sp)
    n = np.sqrt(dcm[0, 0]**2 + dcm[0, 1]**2 + dcm[1, 2]**2 + dcm[2, 2]**2)
    pb = np.arccos(n/np.sqrt(2))
    p = (1.0 - abs(sp))*pa + sp*pb

    return r, p, y


def rpy_to_quat(r, p, y):
    """
    Convert roll, pitch, and yaw to a quaternion, `quat`, vector.  Both
    represent the same right-handed frame rotations.

    Parameters
    ----------
    r : float or int or (N,) np.ndarray
        Roll angles in radians.
    p : float or int or (N,) np.ndarray
        Pitch angles in radians.
    y : float or int or (N,) np.ndarray
        Yaw angles in radians.

    Returns
    -------
    quat : (4,) np.ndarray or (4, N) np.ndarray
        The quaternion vector or a matrix of such vectors.

    See Also
    --------
    quat_to_rpy

    Notes
    -----
    An example use case is to calculate a quaternion that rotates from the
    [nose, right wing, down] body frame to the [north, east, down] navigation
    frame when given a yaw-pitch-roll (z-y-x) frame rotation.

    This function makes sure that the first element of the quaternion is always
    positive.

    The equations to calculate the quaternion are ::

        h = cr cp cy + sr sp sy
        a = sgn(h) h
        b = sgn(h) (sr cp cy - cr sp sy)
        c = sgn(h) (cr sp cy + sr cp sy)
        d = sgn(h) (cr cp sy - sr sp cy)

    where the quaternion is [`a`, `b`, `c`, `d`], the `c` and `s` prefixes
    represent cosine and sine, respectively, the `r`, `p`, and `y` suffixes
    represent roll, pitch, and yaw, respectively, and `sgn` is the sign
    function.  The sign of `h` is used to make sure that the first element of
    the quaternion is always positive.  This is simply a matter of convention.

    References
    ----------
    .. [1]  Titterton & Weston, "Strapdown Inertial Navigation Technology"
    """

    # Depending on the types of inputs,
    if isinstance(r, (float, int)) and isinstance(p, (float, int)) and \
            isinstance(y, (float, int)):
        # Get the half-cosine and half-sines of roll, pitch, and yaw.
        cr = np.cos(r/2.0)
        cp = np.cos(p/2.0)
        cy = np.cos(y/2.0)
        sr = np.sin(r/2.0)
        sp = np.sin(p/2.0)
        sy = np.sin(y/2.0)

        # Build the quaternion vector.
        h = cr*cp*cy + sr*sp*sy
        sgn_h = np.sign(h)
        quat = np.array([sgn_h*h,
                sgn_h*(sr*cp*cy - cr*sp*sy),
                sgn_h*(cr*sp*cy + sr*cp*sy),
                sgn_h*(cr*cp*sy - sr*sp*cy)])
    elif isinstance(r, np.ndarray) and (r.ndim == 1) and \
            isinstance(p, np.ndarray) and (p.ndim == 1) and \
            isinstance(y, np.ndarray) and (y.ndim == 1) and \
            (len(r) == len(p)) and (len(p) == len(y)):
        # Get the half-cosine and half-sines of roll, pitch, and yaw.
        cr = np.cos(r/2.0)
        cp = np.cos(p/2.0)
        cy = np.cos(y/2.0)
        sr = np.sin(r/2.0)
        sp = np.sin(p/2.0)
        sy = np.sin(y/2.0)

        # Build the matrix of quaternion vectors.
        quat = np.zeros((4, len(r)))
        h = cr*cp*cy + sr*sp*sy
        sgn_h = np.sign(h)
        quat[0, :] = sgn_h*h
        quat[1, :] = sgn_h*(sr*cp*cy - cr*sp*sy)
        quat[2, :] = sgn_h*(cr*sp*cy + sr*cp*sy)
        quat[3, :] = sgn_h*(cr*cp*sy - sr*sp*cy)
    else:
        raise Exception('rpy_to_quat: r, p, and y must be floats or ' +
                '(N,) np.ndarrays of equal lengths')

    return quat


def quat_to_rpy(quat):
    """
    Convert from a quaternion right-handed frame rotation to a roll, pitch, and
    yaw, z-y-x sequence of right-handed frame rotations.  If frame 1 is rotated
    in a z-y-x sequence to become frame 2, then the quaternion `quat` would also
    rotate a vector in frame 1 into frame 2.

    Parameters
    ----------
    quat : (4,) np.ndarray or (4, N) np.ndarray
        A quaternion vector or a matrix of such vectors.

    Returns
    -------
    r : float or (N,) np.ndarray
        Roll angles in radians.
    p : float or (N,) np.ndarray
        Pitch angles in radians.
    y : float or (N,) np.ndarray
        Yaw angles in radians.

    See Also
    --------
    rpy_to_quat

    Notes
    -----
    An example use case is the calculation a yaw-roll-pitch (z-y-x) frame
    rotation when given the quaternion that rotates from the [nose, right wing,
    down] body frame to the [north, east, down] navigation frame.

    From the dcm_to_rpy function, we know that the roll, `r`, pitch, `p`, and
    yaw, `y`, can be calculated as follows::

        r = atan2(d23, d33)
        p = -asin(d13)
        y = atan2(d12, d11)

    where the `d` variables are elements of the DCM.  We also know from the
    quat_to_dcm function that ::

              .-                                                            -.
              |   2    2    2    2                                           |
              | (a  + b  - c  - d )    2 (b c + a d)       2 (b d - a c)     |
              |                                                              |
              |                       2    2    2    2                       |
        Dcm = |    2 (b c - a d)    (a  - b  + c  - d )    2 (c d + a b)     |
              |                                                              |
              |                                           2    2    2    2   |
              |    2 (b d + a c)       2 (c d - a b)    (a  - b  - c  + d )  |
              '-                                                            -'

    This means that the `d` variables can be defined in terms of the quaternion
    elements::

               2    2    2    2
        d11 = a  + b  - c  - d           d12 = 2 (b c + a d)

                                         d13 = 2 (b d - a c)
               2    2    2    2
        d33 = a  - b  - c  + d           d23 = 2 (c d + a b)

    This function does not take advantage of the more advanced formula for pitch
    because testing showed it did not help in this case.

    References
    ----------
    .. [1]  Titterton & Weston, "Strapdown Inertial Navigation Technology"
    """

    # Check inputs.
    if not isinstance(quat, np.ndarray) or (quat.ndim > 2):
        raise Exception('quat_to_rpy: quat must be an np.ndarray ' +
                'of less than 3 dimensions')

    # Depending on the dimensions of the input,
    if quat.ndim == 1:
        # Get the required elements of the DCM.
        d11 = quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2
        d12 = 2*(quat[1]*quat[2] + quat[0]*quat[3])
        d13 = 2*(quat[1]*quat[3] - quat[0]*quat[2])
        d23 = 2*(quat[2]*quat[3] + quat[0]*quat[1])
        d33 = quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2

        # Build the output.
        rpy = np.zeros(3)
        rpy[0] = np.arctan2(d23, d33)
        rpy[1] = -np.arcsin(d13)
        rpy[2] = np.arctan2(d12, d11)
    else:
        # Get the required elements of the DCM.
        d11 = quat[0, :]**2 + quat[1, :]**2 - quat[2, :]**2 - quat[3, :]**2
        d12 = 2*(quat[1, :]*quat[2, :] + quat[0, :]*quat[3, :])
        d13 = 2*(quat[1, :]*quat[3, :] - quat[0, :]*quat[2, :])
        d23 = 2*(quat[2, :]*quat[3, :] + quat[0, :]*quat[1, :])
        d33 = quat[0, :]**2 - quat[1, :]**2 - quat[2, :]**2 + quat[3, :]**2

        # Build the output.
        rpy = np.zeros((3, quat.shape[1]))
        rpy[0, :] = np.arctan2(d23, d33)
        rpy[1, :] = -np.arcsin(d13)
        rpy[2, :] = np.arctan2(d12, d11)

    return rpy


def dcm_to_quat(dcm):
    """
    Convert a direction cosine matrix, `dcm`, to a quaternion vector, `quat`.
    Here, the `dcm` is considered to represent a z-y-x sequence of right-handed
    rotations.  This means it has the same sense as the quaternion.

    The implementation here is Cayley's method for obtaining the quaternion.  It
    is used because of its superior numerical accuracy.  This comes from the
    fact that it uses all nine of the elements of the DCM matrix.  It also does
    not suffer from numerical instability due to division as some other methods
    do.

    Parameters
    ----------
    dcm : (3, 3) np.ndarray
        Rotation direction cosine matrix.

    Returns
    -------
    quat : (4,) np.ndarray
        The quaternion vector.

    See Also
    --------
    quat_to_dcm

    Notes
    -----
    FIXME

    References
    ----------
    .. [1]  Titterton & Weston, "Strapdown Inertial Navigation Technology"
    .. [2]  Soheil Sarabandi and Federico Thomas, "A Survey on the Computation
            of Quaternions from Rotation Matrices"
    """

    # Ensure the input is a 3-by-3 np.ndarray.
    if not is_square(dcm, 3):
        raise Exception('dcm_to_quat: dcm must be a (3, 3) np.ndarray')
    if not is_ortho(dcm):
        raise Exception('dcm_to_quat: dcm must be orthogonal')

    # Parse the elements of dcm.
    d00 = dcm[0, 0]
    d01 = dcm[0, 1]
    d02 = dcm[0, 2]
    d10 = dcm[1, 0]
    d11 = dcm[1, 1]
    d12 = dcm[1, 2]
    d20 = dcm[2, 0]
    d21 = dcm[2, 1]
    d22 = dcm[2, 2]

    # Get the squared sums and differences of off-diagonal pairs.
    p01 = (d01 + d10)**2
    p12 = (d12 + d21)**2
    p20 = (d20 + d02)**2
    m01 = (d01 - d10)**2
    m12 = (d12 - d21)**2
    m20 = (d20 - d02)**2

    # Get the magnitudes.
    n0 = np.sqrt((d00 + d11 + d22 + 1)**2 + m12 + m20 + m01)
    n1 = np.sqrt(m12 + (d00 - d11 - d22 + 1)**2 + p01 + p20)
    n2 = np.sqrt(m20 + p01 + (d11 - d00 - d22 + 1)**2 + p12)
    n3 = np.sqrt(m01 + p20 + p12 + (d22 - d00 - d11 + 1)**2)

    # Build the quaternion output.
    quat = 0.25*np.array([n0, np.sign(d12 - d21)*n1,
            np.sign(d20 - d02)*n2, np.sign(d01 - d10)*n3])

    return quat


def quat_to_dcm(quat):
    """
    Convert from a quaternion, `quat`, that performs a right-handed frame
    rotation from frame 1 to frame 2 to a direction cosine matrix, `dcm`, that
    also performs a right-handed frame rotation from frame 1 to frame 2.  The
    `dcm` represents a z-y-x sequence of right-handed rotations.

    Parameters
    ----------
    quat : 4-element 1D np.ndarray
        The 4-element quaternion vector corresponding to the DCM.

    Returns
    -------
    dcm : float 3x3 np.ndarray
        3-by-3 rotation direction cosine matrix.

    See Also
    --------
    dcm_to_quat

    Notes
    -----
    An example use case is to calculate a direction cosine matrix that rotates
    from the [nose, right wing, down] body frame to the [north, east, down]
    navigation frame when given a quaternion frame rotation that rotates from
    the [nose, right wing, down] body frame to the [north, east, down]
    navigation frame.

    The DCM can be defined in terms of the elements of the quaternion
    [a, b, c, d] as ::

              .-                                                            -.
              |   2    2    2    2                                           |
              | (a  + b  - c  - d )    2 (b c + a d)       2 (b d - a c)     |
              |                                                              |
              |                       2    2    2    2                       |
        dcm = |    2 (b c - a d)    (a  - b  + c  - d )    2 (c d + a b)     |
              |                                                              |
              |                                           2    2    2    2   |
              |    2 (b d + a c)       2 (c d - a b)    (a  - b  - c  + d )  |
              '-                                                            -'

    References
    ----------
    .. [1]  Titterton & Weston, "Strapdown Inertial Navigation Technology"
    """

    # Ensure the input is a 4-element np.ndarray.
    if not isinstance(quat, np.ndarray) or (quat.shape != (4,)):
        raise Exception('quat_to_dcm: quat must be a 4-element np.ndarray')

    # Square the elements of the quaternion.
    q0 = quat[0]*quat[0]
    q1 = quat[1]*quat[1]
    q2 = quat[2]*quat[2]
    q3 = quat[3]*quat[3]

    # Build the DCM.
    dcm = np.array([
        [q0 + q1 - q2 - q3,
            2*(quat[1]*quat[2] + quat[0]*quat[3]),
            2*(quat[1]*quat[3] - quat[0]*quat[2])],
        [2*(quat[1]*quat[2] - quat[0]*quat[3]),
            q0 - q1 + q2 - q3,
            2*(quat[2]*quat[3] + quat[0]*quat[1])],
        [2*(quat[1]*quat[3] + quat[0]*quat[2]),
            2*(quat[2]*quat[3] - quat[0]*quat[1]),
            q0 - q1 - q2 + q3]])

    return dcm


def rpy_to_axis_angle(r, p, y):
    """
    Convert roll, pitch, and yaw Euler angles to rotation axis vector and
    rotation angle.

    Parameters
    ----------
    r : float or (N,) np.ndarray
        Roll angles in radians.
    p : float or (N,) np.ndarray
        Pitch angles in radians.
    y : float or (N,) np.ndarray
        Yaw angles in radians.

    Returns
    -------
    ax : (3,) np.ndarray or (3, N) np.ndarray
        Axis vector or matrix of vectors.
    ang : float or (N,) np.ndarray
        Rotation angles in radians.

    See Also
    --------
    axis_angle_to_rpy

    Notes
    -----
    Both (r, p, y) and (ax, ang) represent the same z, y, x sequence of
    right-handed frame rotations.  The conversion happens through an
    intermediate step of calculating the quaternion.  This function makes sure
    that the first element of the quaternion is always positive.

    The equations to calculate the quaternion are ::

        h = cr cp cy + sr sp sy
        a = sgn(h) h
        b = sgn(h) (sr cp cy - cr sp sy)
        c = sgn(h) (cr sp cy + sr cp sy)
        d = sgn(h) (cr cp sy - sr sp cy)

    where the quaternion is `[a, b, c, d]`, the `c` and `s` prefixes
    represent cosine and sine, respectively, the `r`, `p`, and `y` suffixes
    represent roll, pitch, and yaw, respectively, and `sgn` is the sign
    function.  The sign of `h` is used to make sure that the first element of
    the quaternion is always positive.  This is simply a matter of convention.

    Defining the rotation axis vector to be a unit vector, we will define the
    quaterion, `quat`, in terms of the axis and angle:

                  .-     -.               .-     -.              .-   -.
                  |  ang  |               |  ang  |              |  x  |
          a = cos | ----- |     b = x sin | ----- |         ax = |  y  |
                  |   2   |               |   2   |              |  z  |
                  '-     -'               '-     -'              '-   -'
                  .-     -.               .-     -.              .-   -.
                  |  ang  |               |  ang  |              |  a  |
        c = y sin | ----- |     d = z sin | ----- |       quat = |  b  | ,
                  |   2   |               |   2   |              |  c  |
                  '-     -'               '-     -'              |  d  |
                                                                 '-   -'

    Then, the norm of `[b, c, d]` will be

          .-------------     .-----------------------------
         /  2    2    2     /  2    2    2     2 .- ang -.   |     .- ang -. |
        V  b  + c  + d  =  / (x  + y  + z ) sin  | ----- | = | sin | ----- | | .
                          V                      '-  2  -'   |     '-  2  -' |

    Since `a = cos(ang/2)`, with the above value we can get the angle by ::

                        .-  .------------   -.
                        |  /  2    2    2    |
        ang = 2 s atan2 | V  b  + c  + d , a | ,
                        '-                  -'

    where `s` is the sign of the angle determined based on whether the dot
    product of the vector `[b, c, d]` with `[1, 1, 1]` is positive:

        s = sign( b + c + d ) .

    Finally, the axis is calculated by using the first set of equations above:

                  b                     c                     d
        x = -------------     y = -------------     z = ------------- .
                .- ang -.             .- ang -.             .- ang -.
            sin | ----- |         sin | ----- |         sin | ----- |
                '-  2  -'             '-  2  -'             '-  2  -'

    It is true that `ang` and, therefore `sin(ang/2)`, could become 0, which
    would create a singularity.  But, this will happen only if the norm of `[b,
    c, d]` is zero.  In other words, if the quaternion is a vector with only one
    non-zero value, then we will have a problem.

    References
    ----------
    .. [1]  Titterton & Weston, "Strapdown Inertial Navigation Technology"
    """

    # Depending on the dimensions of the inputs,
    if isinstance(r, (float, int)) and isinstance(p, (float, int)) and \
            isinstance(y, (float, int)):
        # Get the cosines and sines of the half angles.
        cr = np.cos(r/2.0)
        sr = np.sin(r/2.0)
        cp = np.cos(p/2.0)
        sp = np.sin(p/2.0)
        cy = np.cos(y/2.0)
        sy = np.sin(y/2.0)

        # Build the quaternion.
        h = cr*cp*cy + sr*sp*sy
        sgn_h = np.sign(h)
        q1 = sgn_h*h
        q2 = sgn_h*(sr*cp*cy - cr*sp*sy)
        q3 = sgn_h*(cr*sp*cy + sr*cp*sy)
        q4 = sgn_h*(cr*cp*sy - sr*sp*cy)

        # Get the norm and sign of the last three elements of the quaternion.
        ax_norm = np.sqrt(q2**2 + q3**2 + q4**2)
        s = np.sign(q2 + q3 + q4)

        # Get the angle of rotation.
        ang = 2*s*np.arctan2(ax_norm, q1)

        # Build the rotation axis vector.
        ax = np.zeros(3)
        k = 1/np.sin(ang/2)
        ax[0] = q2*k
        ax[1] = q3*k
        ax[2] = q4*k

    elif isinstance(r, np.ndarray) and (r.ndim == 1) and \
            isinstance(p, np.ndarray) and (p.ndim == 1) and \
            isinstance(y, np.ndarray) and (y.ndim == 1) and \
            (len(r) == len(p)) and (len(p) == len(y)):
        # Get the cosines and sines of the half angles.
        cr = np.cos(r/2.0)
        sr = np.sin(r/2.0)
        cp = np.cos(p/2.0)
        sp = np.sin(p/2.0)
        cy = np.cos(y/2.0)
        sy = np.sin(y/2.0)

        # Build the quaternion.
        h = cr*cp*cy + sr*sp*sy
        sgn_h = np.sign(h)
        q1 = sgn_h*h
        q2 = sgn_h*(sr*cp*cy - cr*sp*sy)
        q3 = sgn_h*(cr*sp*cy + sr*cp*sy)
        q4 = sgn_h*(cr*cp*sy - sr*sp*cy)

        # Get the norm and sign of the last three elements of the quaternion.
        ax_norm = np.sqrt(q2**2 + q3**2 + q4**2)
        s = np.sign(q2 + q3 + q4)

        # Get the angle of rotation.
        ang = 2*s*np.arctan2(ax_norm, q1)

        # Build the rotation axis vector.
        ax = np.zeros((3, len(r)))
        k = 1/np.sin(ang/2)
        ax[0, :] = q2*k
        ax[1, :] = q3*k
        ax[2, :] = q4*k
    else:
        raise Exception('rpy_to_axis_angle: r, p, and y must be scalars or ' +
                '(N,) np.ndarrays of equal lengths.')

    return ax, ang


# FIXME add axis_angle_to_rpy
# FIXME add dcm_to_axis_angle
# FIXME add axis_angle_to_dcm
# FIXME add quat_to_axis_angle
# FIXME add axis_angle_to_quat


def axis_angle_to_dcm(ax, ang):
    """
    Create a direction cosine matrix (DCM) (also known as a rotation matrix) to
    rotate from one frame to another given a rotation `ax` vector and a
    right-handed `ang` of rotation.

    Parameters
    ----------
    ax : array_like
        Vector of the rotation axis with three values.
    ang : float
        Rotation ang in radians.

    Returns
    -------
    R : 2D np.ndarray
        3x3 rotation matrix.
    """

    # Normalize the rotation axis vector.
    ax = ax/np.norm(ax)

    # Parse the rotation axis vector into its three elements.
    x = ax[0]
    y = ax[1]
    z = ax[2]

    # Get the cosine and sine of the ang.
    co = np.cos(ang)
    si = np.sin(ang)

    # Build the direction cosine matrix.
    R = np.array([
        [ co + x**2*(1 - co), x*y*(1 - co) - z*si, x*z*(1 - co) + y*si],
        [y*x*(1 - co) + z*si,  co + y**2*(1 - co), y*z*(1 - co) - x*si],
        [z*x*(1 - co) - y*si, z*y*(1 - co) + x*si,  co + z**2*(1 - co)]])

    return R


def rot(ang, ax=2, degrees=False):
    """
    Build a three-dimensional rotation matrix of rotation angle `ang` about
    the axis `ax`.

    Parameters
    ----------
    ang : float or int or array_like
        Angle of rotation in radians (or degrees if `degrees` is True).
    ax : {0, 1, 2}, float or int or array_like, default 2
        Axis about which to rotate.
    degrees : bool, default False
        A flag denoting whether the values of `ang` are in degrees.

    See Also
    --------
    rpy_to_dcm

    Returns
    -------
    R : 2D np.ndarray
        3x3 rotation matrix
    """

    # Control the input types.
    if isinstance(ang, (float, int)):
        ang = np.array([float(ang)])
    elif isinstance(ang, list):
        ang = np.array(ang, dtype=float)
    if isinstance(ax, (float, int)):
        ax = np.array([int(ax)])
    elif isinstance(ax, list):
        ax = np.array(ax, dtype=int)

    # Check the lengths of ang and ax.
    if len(ang) != len(ax):
        raise Exception("rot: ang and ax must be the same length!")
    else:
        N = len(ang)

    # Convert degrees to radians.
    if degrees:
        ang *= DEG_TO_RAD

    # Build the rotation matrix.
    R = np.eye(3)
    for n in range(N):
        # Skip trivial rotations.
        if ang[n] == 0:
            continue

        # Get the cosine and sine of this ang.
        co = np.cos(ang[n])
        si = np.sin(ang[n])

        # Pre-multiply by another matrix.
        if ax[n] == 0:
            R = np.array([[1, 0, 0], [0, co, si], [0, -si, co]]).dot(R)
        elif ax[n] == 1:
            R = np.array([[co, 0, -si], [0, 1, 0], [si, 0, co]]).dot(R)
        elif ax[n] == 2:
            R = np.array([[co, si, 0], [-si, co, 0], [0, 0, 1]]).dot(R)
        else:
            raise Exception("rot: Axis must be 0 to 2.")

    return R


def ecef_to_geodetic(xe, ye, ze):
    """
    Convert an ECEF (Earth-centered, Earth-fixed) position to geodetic
    coordinates.  This follows the WGS-84 definitions (see WGS-84 Reference
    System (DMA report TR 8350.2)).

    Parameters
    ----------
    xe, ye, ze : float
        ECEF x, y, and z-axis position values in meters.

    Returns
    -------
    phi : float
        Geodetic latitude in radians.
    lam : float
        Geodetic longitude in radians.
    hae : float
        Height above ellipsoid in meters.

    See Also
    --------
    geodetic_to_ecef

    Notes
    -----
    Note that inherent in solving the problem of getting the geodetic latitude
    and ellipsoidal height is finding the roots of a quartic polynomial because
    we are looking for the intersection of a line with an ellipse.  While there
    are closed-form solutions to this problem (see Wikipedia), each point has
    potentially four solutions and the solutions are not numerically stable.
    Instead, this function uses the Newton-Raphson method to iteratively solve
    for the geodetic coordinates.

    First, we want to approximate the values for geodetic latitude, `phi`, and
    height above ellipsoid, `hae`, given the (x, y, z) position in the ECEF
    frame::

                                .------
                               / 2    2
        hae = 0         rho = V x  + y             phi = atan2(z, rho),

    where `rho` is the distance from the z axis of the ECEF frame.  (While there
    are better approximations for `hae` than zero, the improvement in accuracy
    was not enough to reduce the number of iterations and the additional
    computational burden could not be justified.)  Then, we will iteratively use
    this approximation for `phi` and `hae` to calculate what `rho` and `z` would
    be, get the residuals given the correct `rho` and `z` values in the ECEF
    frame, use the inverse Jacobian to calculate the corresponding residuals of
    `phi` and `hae`, and update our approximations for `phi` and `hae` with
    those residuals.  In testing millions of randomly generated points, three
    iterations was sufficient to reach the limit of numerical precision for
    64-bit floating-point numbers.

    So, first, let us define the transverse, `Rt`, and meridional, `Rm`, radii::

                                   .-        -.               .--------------
              a                a   |       2  |              /     2   2
        Rt = ----       Rm = ----- |  1 - e   |     kphi =  V 1 - e sin (phi) ,
             kphi                3 |          |
                             kphi  '-        -'

    where `e` is the eccentricity of the Earth, and `a` is the semi-major radius
    of the Earth.  The ECEF-frame `rho` and `z` values given the approximations
    to geodetic latitude, `phi`, and height above ellipsoid, `hae`, are ::

         ~                              ~                    2
        rho = cos(phi) (Rt + hae)       z = sin(phi) (Rm kphi + hae) .

    We already know the correct values for `rho` and `z`, so we can get
    residuals::

                      ~                         ~
        drho = rho - rho               dz = z - z .

    We can relate the `rho` and `z` residuals to the `phi` and `hae` residuals
    by using the inverse Jacobian matrix::

        .-    -.       .-    -.
        | dphi |    -1 | drho |
        |      | = J   |      | .
        | dhae |       |  dz  |
        '-    -'       '-    -'

    With a bit of algebra, we can combine and simplify the calculation of the
    Jacobian with the calculation of the `phi` and `hae` residuals::

        dhae = ( c*drho + s*dz)
        dphi = (-s*drho + c*dz)/(Rm + hae) .

    Conceptually, this is the backwards rotation of the (`drho`, `dz`) vector by
    the angle `phi`, where the resulting y component of the rotated vector is
    treated as an arc length and converted to an angle, `dphi`, using the radius
    `Rm` + `hae`.  With the residuals for `phi` and `hae`, we can update our
    approximations for `phi` and `hae`::

        phi = phi + dphi
        hae = hae + dhae

    and iterate again.  Finally, the longitude, `lam`, is exactly the arctangent
    of the ECEF `x` and `y` values::

        lam = atan2(y, x) .

    References
    ----------
    .. [1]  WGS-84 Reference System (DMA report TR 8350.2)
    .. [2]  Inertial Navigation: Theory and Implementation by David Woodburn and
            Robert Leishman
    """

    # Reform the inputs to ndarrays of floats.
    x = np.asarray(xe).astype(float)
    y = np.asarray(ye).astype(float)
    z = np.asarray(ze).astype(float)

    # Initialize the height above the ellipsoid.
    hae = 0

    # Get the true radial distance from the z axis.
    rho = np.sqrt(x**2 + y**2)

    # Initialize the estimated ground latitude.
    phi = np.arctan2(z, rho) # bound to [-pi/2, pi/2]

    # Iterate to reduce residuals of the estimated closest point on the ellipse.
    for _ in range(3):
        # Using the estimated ground latitude, get the cosine and sine.
        co = np.cos(phi)
        si = np.sin(phi)
        kphi2 = 1 - E2*si**2
        kphi = np.sqrt(kphi2)
        Rt = A_E/kphi
        Rm = A_E*(1 - E2)/(kphi*kphi2)

        # Get the estimated position in the meridional plane (the plane defined
        # by the longitude and the z axis).
        rho_est = co*(Rt + hae)
        z_est = si*(Rm*kphi2 + hae)

        # Get the residuals.
        drho = rho - rho_est
        dz = z - z_est

        # Using the inverse Jacobian, get the residuals in phi and hae.
        dphi = (co*dz - si*drho)/(Rm + hae)
        dhae = (si*dz + co*drho)

        # Adjust the estimated ground latitude and ellipsoidal height.
        phi = phi + dphi
        hae = hae + dhae

    # Get the longitude.
    lam = np.arctan2(y, x)

    # Reduce arrays of length 1 to scalars.
    if phi.size == 1:
        phi = phi.item()
        lam = lam.item()
        hae = hae.item()

    return phi, lam, hae


def geodetic_to_ecef(phi, lam, hae):
    """
    Convert position in geodetic coordinates to ECEF (Earth-centered,
    Earth-fixed) coordinates.  This method is direct and not an approximation.
    This follows the WGS-84 definitions (see WGS-84 Reference System (DMA report
    TR 8350.2)).

    Parameters
    ----------
    phi : float or array_like
        Geodetic latitude in radians.
    lam : float or array_like
        Geodetic longitude in radians.
    hae : float or array_like
        Height above ellipsoid in meters.

    Returns
    -------
    xe, ye, ze : float or array_like
        ECEF x, y, and z-axis position values in meters.

    See Also
    --------
    ecef_to_geodetic

    Notes
    -----
    The distance from the z axis is ::

              .-  a        -.
        rho = |  ---- + hae | cos(phi)
              '- kphi      -'

    where `a` is the semi-major radius of the earth and ::

                  .---------------
                 /     2    2
        kphi =  V 1 - e  sin (phi)
                       E

    The `e sub E` value is the eccentricity of the earth.  Knowing the distance
    from the z axis, we can get the x and y coordinates::

         e                       e
        x  = rho cos(lam)       y  = rho sin(lam) .

    The z-axis coordinate is ::

         e   .-  a         2        -.
        z  = |  ---- (1 - e ) + hae  | sin(phi) .
             '- kphi       E        -'

    Several of these equations are admittedly not intuitively obvious.  The
    interested reader should refer to external texts for insight.

    References
    ----------
    .. [1]  WGS-84 Reference System (DMA report TR 8350.2)
    .. [2]  Inertial Navigation: Theory and Implementation by David Woodburn and
            Robert Leishman
    """

    # Reform the inputs to ndarrays of floats.
    phi = np.asarray(phi).astype(float)
    lam = np.asarray(lam).astype(float)
    hae = np.asarray(hae).astype(float)

    # Get the distance from the z axis.
    kphi = np.sqrt(1 - E2*np.sin(phi)**2)
    rho = (A_E/kphi + hae)*np.cos(phi)

    # Get the x, y, and z coordinates.
    xe = rho*np.cos(lam)
    ye = rho*np.sin(lam)
    ze = (A_E/kphi*(1 - E2) + hae)*np.sin(phi)

    # Reduce arrays of length 1 to scalars.
    if xe.size == 1:
        xe = xe.item()
        ye = ye.item()
        ze = ze.item()

    return xe, ye, ze


def ecef_to_tangent(xe, ye, ze, xe0=None, ye0=None, ze0=None, ned=True):
    """
    Convert ECEF (Earth-centered, Earth-fixed) coordinates, with a defined local
    origin, to local, tangent Cartesian North, East, Down (NED) or East, North,
    Up (ENU) coordinates.

    Parameters
    ----------
    xe, ye, ze : float or array_like
        ECEF x, y, and z-axis position values in meters.
    xe0, ye0, ze0 : float, default 0
        ECEF x, y, and z-axis origin values in meters.
    ned : bool, default True
        Flag to use NED (True) or ENU (False) orientation.

    Returns
    -------
    xt, yt, zt : float or array_like
        Local, tanget x, y, and z-axis position values in meters.

    See Also
    --------
    tangent_to_ecef

    Notes
    -----
    First, the ECEF origin is converted to geodetic coordinates.  Then, those
    coordinates are used to calculate a rotation matrix from the ECEF frame to
    the local, tangent Cartesian frame::

             .-                     -.
         n   |  -sp cl  -sp sl   cp  |
        R  = |    -sl     cl      0  |      NED
         e   |  -cp cl  -cp sl  -sp  |
             '-                     -'

             .-                     -.
         n   |    -sl     cl      0  |
        R  = |  -sp cl  -sp sl   cp  |      ENU
         e   |   cp cl   cp sl   sp  |
             '-                     -'

    where `sp` and `cp` are the sine and cosine of the origin latitude,
    respectively, and `sl` and `cl` are the sine and cosine of the origin
    longitude, respectively.  Then, the displacement vector of the ECEF position
    relative to the ECEF origin is rotated into the local, tangent frame::

        .-  -.      .-        -.
        | xt |    n | xe - xe0 |
        | yt | = R  | ye - ye0 | .
        | zt |    e | ze - ze0 |
        '-  -'      '-        -'

    If `xe0`, `ye0`, and `ze0` are not provided (or are all zeros), the first
    values of `xe`, `ye`, and `ze` will be used as the origin.
    """

    # Reform the inputs to ndarrays of floats.
    xe = np.asarray(xe).astype(float)
    ye = np.asarray(ye).astype(float)
    ze = np.asarray(ze).astype(float)

    # Use the first point as the origin if otherwise not provided.
    if (xe0 == None) and (ye0 == None) and (ze0 == None):
        xe0 = xe[0]
        ye0 = ye[0]
        ze0 = ze[0]

    # Get the local-level coordinates.
    phi0, lam0, _ = ecef_to_geodetic(xe0, ye0, ze0)

    # Get the cosines and sines of the latitude and longitude.
    cp = np.cos(phi0)
    sp = np.sin(phi0)
    cl = np.cos(lam0)
    sl = np.sin(lam0)

    # Get the displacement ECEF vector from the origin.
    dxe = xe - xe0
    dye = ye - ye0
    dze = ze - ze0

    # Get the local, tangent coordinates.
    if ned:
        xt = -sp*cl*dxe - sp*sl*dye + cp*dze
        yt =    -sl*dxe +    cl*dye
        zt = -cp*cl*dxe - cp*sl*dye - sp*dze
    else:
        xt =    -sl*dxe +    cl*dye
        yt = -sp*cl*dxe - sp*sl*dye + cp*dze
        zt =  cp*cl*dxe + cp*sl*dye + sp*dze

    # Reduce arrays of length 1 to scalars.
    if xt.size == 1:
        xt = xt.item()
        yt = yt.item()
        zt = zt.item()

    return xt, yt, zt


def tangent_to_ecef(xt, yt, zt, xe0, ye0, ze0, ned=True):
    """
    Convert local, tangent Cartesian North, East, Down (NED) or East, North, Up
    (ENU) coordinates, with a defined local origin, to ECEF (Earth-centered,
    Earth-fixed) coordinates.

    Parameters
    ----------
    xt, yt, zt : float or array_like
        Local, tanget x, y, and z-axis position values in meters.
    xe0, ye0, ze0 : float, default 0
        ECEF x, y, and z-axis origin values in meters.
    ned : bool, default True
        Flag to use NED or ENU orientation.

    Returns
    -------
    xe, ye, ze : float or array_like
        ECEF x, y, and z-axis position values in meters.

    See Also
    --------
    ecef_to_tangent

    Notes
    -----
    First, the ECEF origin is converted to geodetic coordinates.  Then, those
    coordinates are used to calculate a rotation matrix from the ECEF frame to
    the local, tangent Cartesian frame::

             .-                     -.
         e   |  -sp cl  -sl  -cp cl  |
        R  = |  -sp sl   cl  -cp sl  |      NED
         n   |    cp      0   -sp    |
             '-                     -'

             .-                     -.
         e   |   -sl  -sp cl  cp cl  |
        R  = |    cl  -sp sl  cp sl  |      ENU
         n   |     0    cp     sp    |
             '-                     -'

    where `sp` and `cp` are the sine and cosine of the origin latitude,
    respectively, and `sl` and `cl` are the sine and cosine of the origin
    longitude, respectively.  Then, the displacement vector of the ECEF position
    relative to the ECEF origin is rotated into the local, tangent frame::

        .-  -.      .-  -.   .-   -.
        | xe |    e | xt |   | xe0 |
        | ye | = R  | yt | + | ye0 | .
        | ze |    n | zt |   | ze0 |
        '-  -'      '-  -'   '-   -'

    The scalars `xe0`, `ye0`, and `ze0` defining the origin must be given and
    cannot be inferred.
    """

    # Reform the inputs to ndarrays of floats.
    xt = np.asarray(xt).astype(float)
    yt = np.asarray(yt).astype(float)
    zt = np.asarray(zt).astype(float)

    # Get the local-level coordinates.
    phi0, lam0, _ = ecef_to_geodetic(xe0, ye0, ze0)

    # Get the cosines and sines of the latitude and longitude.
    cp = np.cos(phi0)
    sp = np.sin(phi0)
    cl = np.cos(lam0)
    sl = np.sin(lam0)

    # Get the local, tangent coordinates.
    if ned:
        xe = -sp*cl*xt - sl*yt - cp*cl*zt + xe0
        ye = -sp*sl*xt + cl*yt - cp*sl*zt + ye0
        ze =     cp*xt         -    sp*zt + ze0
    else:
        xe = -sl*xt - sp*cl*yt + cp*cl*zt + xe0
        ye =  cl*xt - sp*sl*yt + cp*sl*zt + ye0
        ze =        +    cp*yt +    sp*zt + ze0

    # Reduce arrays of length 1 to scalars.
    if xe.size == 1:
        xe = xe.item()
        ye = ye.item()
        ze = ze.item()

    return xe, ye, ze


def geodetic_to_curlin(phi, lam, hae, phi0=None, lam0=None, hae0=None,
        ned=True):
    """
    Convert geodetic coordinates with a geodetic origin to local, curvilinear
    position in either North, East, Down (NED) or East, North, Up (ENU)
    coordinates.

    Parameters
    ----------
    phi : float or array_like
        Geodetic latitude in radians.
    lam : float or array_like
        Geodetic longitude in radians.
    hae : float or array_like
        Height above ellipsoid in meters.
    phi0 : float, default None
        Geodetic latitude origin in radians.
    lam0 : float, default None
        Geodetic longitude origin in radians.
    hae0 : float, default None
        Heigh above ellipsoid origin in meters.
    ned : bool, default True
        Flag to use NED (True) or ENU (False) orientation.

    Returns
    -------
    xc, yc, zc : float or array_like
        ECEF x, y, and z-axis position values in meters.

    See Also
    --------
    curlin_to_geodetic

    Notes
    -----
    The equations are ::

        .-  -.   .-                                -.
        | xc |   |     (Rm + hae) (phi - phi0)      |
        | yc | = | (Rt + hae) cos(phi) (lam - lam0) |       NED
        | zc |   |           (hae0 - hae)           |
        '-  -'   '-                                -'

    or ::

        .-  -.   .-                                -.
        | xc |   | (Rt + hae) cos(phi) (lam - lam0) |
        | yc | = |     (Rm + hae) (phi - phi0)      |       ENU
        | zc |   |           (hae - hae0)           |
        '-  -'   '-                                -'

    where ::

                                     2
                             a (1 - e )                 .--------------
              a                      E                 /     2   2
        Rt = ----       Rm = ----------       kphi =  V 1 - e sin (lat) .
             kphi                 3                          E
                              kphi

    Here, `a` is the semi-major axis of the Earth, `e sub E` is the eccentricity
    of the Earth, `Rt` is the transverse radius of curvature of the Earth, and
    `Rm` is the meridional radius of curvature of the Earth.

    If `phi0`, `lam0`, and `hae0` are not provided (are left as `None`), the
    first values of `phi`, `lam`, and `hae` will be used as the origin.

    References
    ----------
    .. [1]  Titterton & Weston, "Strapdown Inertial Navigation Technology"
    .. [2]  https://en.wikipedia.org/wiki/Earth_radius#Meridional
    .. [3]  https://en.wikipedia.org/wiki/Earth_radius#Prime_vertical
    """

    # Reform the inputs to ndarrays of floats.
    phi = np.asarray(phi).astype(float)
    lam = np.asarray(lam).astype(float)
    hae = np.asarray(hae).astype(float)

    # Use the first point as the origin if otherwise not provided.
    if (phi0 == None) and (lam0 == None) and (hae0 == None):
        phi0 = phi[0]
        lam0 = lam[0]
        hae0 = hae[0]

    # Get the parallel and meridional radii of curvature.
    kphi = np.sqrt(1 - E2*np.sin(phi)**2)
    Rt = A_E/kphi
    Rm = A_E*(1 - E2)/kphi**3

    # Get the curvilinear coordinates.
    if ned: # NED
        xc = (Rm + hae)*(phi - phi0)
        yc = (Rt + hae)*np.cos(phi)*(lam - lam0)
        zc = hae0 - hae
    else:   # ENU
        xc = (Rt + hae)*np.cos(phi)*(lam - lam0)
        yc = (Rm + hae)*(phi - phi0)
        zc = hae - hae0

    # Reduce arrays of length 1 to scalars.
    if xc.size == 1:
        xc = xc.item()
        yc = yc.item()
        zc = zc.item()

    return xc, yc, zc


def curlin_to_geodetic(xc, yc, zc, phi0, lam0, hae0, ned=True):
    """
    Convert local, curvilinear position in either North, East, Down (NED) or
    East, North, Up (ENU) coordinates to geodetic coordinates with a geodetic
    origin.  The solution is iterative, using the Newton-Raphson method.

    Parameters
    ----------
    xc, yc, zc : float or array_like
        Navigation-frame x, y, and z-axis position values in meters.
    phi0 : float, default 0
        Geodetic latitude origin in radians.
    lam0 : float, default 0
        Geodetic longitude origin in radians.
    hae0 : float, default 0
        Heigh above ellipsoid origin in meters.
    ned : bool, default True
        Flag to use NED (True) or ENU (False) orientation.

    Returns
    -------
    phi : float or array_like
        Geodetic latitude in radians.
    lam : float or array_like
        Geodetic longitude in radians.
    hae : float or array_like
        Height above ellipsoid in meters.

    See Also
    --------
    geodetic_to_curlin

    Notes
    -----
    The equations to get curvilinear coordinates from geodetic are ::

        .-  -.   .-                                -.
        | xc |   |     (Rm + hae) (phi - phi0)      |
        | yc | = | (Rt + hae) cos(phi) (lam - lam0) |       NED
        | zc |   |           (hae0 - hae)           |
        '-  -'   '-                                -'

    or ::

        .-  -.   .-                                -.
        | xc |   | (Rt + hae) cos(phi) (lam - lam0) |
        | yc | = |     (Rm + hae) (phi - phi0)      |       ENU
        | zc |   |           (hae - hae0)           |
        '-  -'   '-                                -'

    where ::

                                     2
                             a (1 - e )                 .--------------
              a                      E                 /     2   2
        Rt = ----       Rm = ----------       kphi =  V 1 - e sin (lat) .
             kphi                 3                          E
                              kphi

    Here, `a` is the semi-major axis of the Earth, `e sub E` is the eccentricity
    of the Earth, `Rt` is the transverse radius of curvature of the Earth, and
    `Rm` is the meridional radius of curvature of the Earth.  Unfortunately, the
    reverse process to get geodetic coordinates from curvilinear coordinates is
    not as straightforward.  So the Newton-Raphson method is used.  Using NED as
    an example, with the above equations, we can write the differential relation
    as follows::

        .-    -.     .-      -.           .-           -.
        |  dx  |     |  dphi  |           |  J11   J12  |
        |      | = J |        |       J = |             | ,
        |  dy  |     |  dlam  |           |  J21   J22  |
        '-    -'     '-      -'           '-           -'

    where the elements of the Jacobian J are ::

              .-    2        -.
              |  3 e  Rm s c  |
        J11 = |     E         | (phi - phi0) + Rm + h
              | ------------- |
              |        2      |
              '-   kphi      -'

        J12 = 0

              .- .-  2  2     -.         -.
              |  |  e  c       |          |
              |  |   E         |          |
        J21 = |  | ------- - 1 | Rt - hae | s (lam - lam0)
              |  |      2      |          |
              '- '- kphi      -'         -'

        J22 = (Rt + hae) c.

    where `s` and `c` are the sine and cosine of `phi`, respectively.  Using the
    inverse Jacobian, we can get the residuals of `phi` and `lam` from the
    residuals of `xc` and `yc`::

                 J22 dx - J12 dy
        dphi = -------------------
                J11 J22 - J21 J12

                 J11 dy - J21 dx
        dlam = ------------------- .
                J11 J22 - J21 J12

    These residuals are added to the estimated `phi` and `lam` values and
    another iteration begins.

    References
    ----------
    .. [1]  Titterton & Weston, "Strapdown Inertial Navigation Technology"
    .. [2]  https://en.wikipedia.org/wiki/Earth_radius#Meridional
    .. [3]  https://en.wikipedia.org/wiki/Earth_radius#Prime_vertical
    """

    # Reform the inputs to ndarrays of floats.
    xc = np.asarray(xc).astype(float)
    yc = np.asarray(yc).astype(float)
    zc = np.asarray(zc).astype(float)

    # Flip the orientation if it is ENU.
    if not ned:
        zc = zc*(-1)
        temp = xc
        xc = yc*1
        yc = temp*1

    # Define height.
    hae = hae0 - zc

    # Initialize the latitude and longitude.
    phi = phi0 + xc/(A_E + hae)
    lam = lam0 + yc/((A_E + hae)*np.cos(phi))

    # Iterate.
    for _ in range(3):
        # Get the sine and cosine of latitude.
        si = np.sin(phi)
        co = np.cos(phi)

        # Get the parallel and meridional radii of curvature.
        kp2 = 1 - E2*si**2
        kphi = np.sqrt(kp2)
        Rt = A_E/kphi
        Rm = A_E*(1 - E2)/kphi**3

        # Get the estimated xy position.
        xce = (Rm + hae)*(phi - phi0)
        yce = (Rt + hae)*co*(lam - lam0)

        # Get the residual.
        dxc = xc - xce
        dyc = yc - yce

        # Get the inverse Jacobian.
        J11 = (3*E2*Rm*si*co/kp2)*(phi - phi0) + Rm + hae
        J12 = 0
        J21 = ((E2*co**2/kp2 - 1)*Rt - hae)*si*(lam - lam0)
        J22 = (Rt + hae)*co
        Jdet_inv = 1/(J11*J22 - J21*J12)

        # Using the inverse Jacobian, get the residuals in phi and lam.
        dphi = (J22*dxc - J12*dyc)*Jdet_inv
        dlam = (J11*dyc - J21*dxc)*Jdet_inv

        # Update the latitude and longitude.
        phi = phi + dphi
        lam = lam + dlam

    # Reduce arrays of length 1 to scalars.
    if phi.size == 1:
        phi = phi.item()
        lam = lam.item()
        hae = hae.item()

    return phi, lam, hae


def somigliana(lat, hae):
    """
    Calculate the scalar component of gravity using the Somigliana equation.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    hae : float
        Height above ellipsoid in meters.

    Returns
    -------
    gh : float
        Scalar component of gravity in meters per second squared.

    References
    ----------
    .. [1]  https://en.wikipedia.org/wiki/Theoretical_gravity
    """

    # Constants
    SG_GE = 9.7803253359                # Somigliana coefficient ge [m/s^2]
    SG_K = 1.93185265241e-3             # Somigliana coefficient k [ND]
    SG_F = 3.35281066475e-3             # Somigliana coefficient f [ND]
    SG_M = 3.44978650684e-3             # Somigliana coefficient m [ND]

    slat = np.sin(lat)
    klat = np.sqrt(1 - E2*slat**2)
    g0 = SG_GE*(1 + SG_K*slat**2)/klat
    gh = g0*(1 + (3/A_E**2)*hae**2 - 2/A_E*(1 + SG_F + SG_M -
            2*SG_F*slat**2)*hae)
    return gh

# --------------------
# Satellite Navigation
# --------------------

def gold10(sv):
    """
    Get the array of the 1023 PRN Gold code chip values given a satellite
    number.  The original design comes from Nathan Bergey.

    Parameters
    ----------
    sv : integer
        Satellite number (1 to 32).

    Returns
    -------
    ca : 1D np.ndarray
        C/A code, array of 1023 1s and -1s.

    References
    ----------
    .. [1]  https://natronics.github.io/blag/2014/gps-prn/.
    """

    # Define the tap indices dictionary for each PRN.
    SV = {
        1:  [1, 5],  2:  [2, 6],  3:  [3, 7],  4:  [4, 8],
        5:  [0, 8],  6:  [1, 9],  7:  [0, 7],  8:  [1, 8],
        9:  [2, 9],  10: [1, 2],  11: [2, 3],  12: [4, 5],
        13: [5, 6],  14: [6, 7],  15: [7, 8],  16: [8, 9],
        17: [0, 3],  18: [1, 4],  19: [2, 5],  20: [3, 6],
        21: [4, 7],  22: [5, 8],  23: [0, 2],  24: [3, 5],
        25: [4, 6],  26: [5, 7],  27: [6, 8],  28: [7, 9],
        29: [0, 5],  30: [1, 6],  31: [2, 7],  32: [3, 8]}

    # Define the shift function.
    def shift(register, feedback, output):
        # Calculate output.
        out = [register[i] for i in output]
        out = sum(out) % 2

        # Sum the select elements of register specified by feedback - 1 and get
        # the modulous of that sum with respect to 2.
        fb = sum([register[i] for i in feedback]) % 2

        # Shift the elements of register to the right.  The last element is
        # lost.  The second element is a duplicate of the first.
        for i in reversed(range(len(register) - 1)):
            register[i + 1] = register[i]

        # Put the feedback (fb) into the first element.
        register[0] = fb

        return out

    # Initialize arrays.
    G1 = [1]*10
    G2 = [1]*10
    ca = np.zeros(1023)

    # Create sequence
    for i in range(1023):
        g1 = shift(G1, [2, 9], [9])
        g2 = shift(G2, [1, 2, 5, 7, 8, 9], SV[sv])

        # Modulo 2 add and append to the code
        ca[i] = 2*((g1 + g2) % 2) - 1

    # Return C/A code.
    return ca


def sv_elevation(x_r, y_r, z_r, x_s, y_s, z_s):
    """
    Get the space vehicle (satellite) elevation angle above the horizon relative
    to the receiver.

    Parameters
    ----------
    x_r, y_r, z_r : float or array_like
        x, y, and z-axis coordinates of a GNSS receiver in meters.
    x_s, y_s, z_s : float or array_like
        x, y, and z-axis coordinates of a GNSS satellite in meters.

    Returns
    -------
    el : float or np.ndarray
        Elevation angle in radians.  Level with the horizon is 0 rad.

    Notes
    -----
    First, it calculates the vector from the receiver to the space vehicle::

        ^   s o
        |    / \                .-    -.   .-   -.   .-   -.
        |   /    \   rs         | x_rs |   | x_s |   | x_r |
        |  /       \            | y_rs | = | y_s | - | y_r |
        | /     ..--o r         | z_rs |   | z_s |   | z_r |
        |/..--``                '-    -'   '-   -'   '-   -'
        o------------->

    Second, it calculates the upward vector, pointing into the sky.  This is
    actually just the <x, y, z> position vector of the receiver.  Both of these
    vectors get normalized by their respective lengths.  Third, it uses the
    definition of the dot product between those first two vectors to get the
    angle of the space vehicle above the horizon.  In calculating the upward
    vector, this function models the earth as a perfect sphere.  The error
    induced by this simplification causes at most 0.19 degrees of error.
    """

    # Reform the inputs to ndarrays of floats.
    x_r = np.asarray(x_r).astype(float)
    y_r = np.asarray(y_r).astype(float)
    z_r = np.asarray(z_r).astype(float)
    x_s = np.asarray(x_s).astype(float)
    y_s = np.asarray(y_s).astype(float)
    z_s = np.asarray(z_s).astype(float)

    # Reform one set to 2D if the other set is 2D.
    if (x_s.ndim == 2) and (x_r.ndim == 1):
        x_r = x_r.reshape(-1, 1)
        y_r = y_r.reshape(-1, 1)
        z_r = z_r.reshape(-1, 1)

    # vector from receiver to space vehicle
    x_rs = x_s - x_r
    y_rs = y_s - y_r
    z_rs = z_s - z_r

    # norm of vector from receiver to space vehicle
    n_rs = np.sqrt(x_rs**2 + y_rs**2 + z_rs**2)

    # normalized vector
    x_rs /= n_rs
    y_rs /= n_rs
    z_rs /= n_rs

    # upward vector based on geodetic coordinates
    n_r = np.sqrt(x_r**2 + y_r**2 + z_r**2)
    x_up = x_r/n_r
    y_up = y_r/n_r
    z_up = z_r/n_r

    # elevation angle
    el = np.arcsin(x_rs*x_up + y_rs*y_up + z_rs*z_up)

    # Reduce arrays of length 1 to scalars.
    if el.size == 1:
        el = el.item()

    return el


def trop_correction(hae, el):
    """
    Model the correction to the pseudorange in order to account for the delay
    caused by the troposphere.

    Parameters
    ----------
    hae : float or array_like
        Height above ellipsoid in meters.
    el : float or array_like
        Elevation angle above the horizon in radians.

    Returns
    -------
    drho : float or array_like
        Correction to the pseudorange in meters.

    Notes
    -----
    Simply add this value to the raw pseudorange in order to correct it.  This
    model was built by creating rational function fits to a far more complicated
    model.  The maximum error is about 7.5 cm and the RMSE is about 1.4 cm over
    a range of 0 to 15 km of height and 5 to 90 degrees of elevation angle.
    """

    k = (1 - 4.3e-5*hae)/(1 + hae*(8.1e-5 + 4e-9*hae))
    num = -0.7704 + el*(-13.832 + el*(1.3314 - el))
    den = 0.01 + el*(0.29643 + el*(5.79263 - el*(0.74224 + 0.3833*el)))
    drho = k*num/den

    return drho

# ---------------------
# Stochastic Estimation
# ---------------------

def vanloan(F, B=None, Q=None, T_Sa=1.0):
    """
    Apply the Van Loan method to the matrices `F`, `B`, and `Q`.

    Parameters
    ----------
    F : 2D np.ndarray
        Continuous-domain dynamics matrix.
    B : 2D np.ndarray, default None
        Continuous-domain dynamics input matrix.  To omit this input, provide
        `None`.
    Q : 2D np.ndarray, default None
        Continuous-domain dynamics noise covariance matrix.  To omit this input,
        provide `None`.
    T_Sa : float, default 1.0
        Sampling period in seconds.

    Returns
    -------
    Phi : 2D np.ndarray
        Discrete-domain dynamics matrix.
    Bd : 2D np.ndarray
        Discrete-domain dynamics input matrix.
    Qd : 2D np.ndarray
        Discrete-domain dynamics noise covariance matrix.

    Notes
    -----
    The Van Loan method, named after Charles Van Loan, is one way of
    discretizing the matrices of a state-space system.  Suppose that you have
    the following state-space system::

        .                 .--
        x = F x + B u +  V Q  w

        y = C x + D u + R v

    where `x` is the state vector, `u` is the input vector, and `w` is a white,
    Gaussian noise vector with means of zero and variances of one.  Then, to get
    the discrete form of this equation, we would need to find `Phi`, `Bd`, and
    `Qd` such that ::

                             .--
        x = Phi x + Bd u +  V Qd w

        y = C x + D u + Rd v

    `Rd` is simply `R`.  `C` and `D` are unaffected by the discretization
    process.  We can find `Phi` and `Qd` by doing the following::

            .-      -.                    .-          -.
            | -F  Q  |                    |  M11  M12  |
        L = |        |    M = expm(L T) = |            |
            |  0  F' |                    |  M21  M22  |
            '-      -'                    '-          -'
        Phi = M22'        Qd = Phi M12 .

    Note that `F` must be square and `Q` must have the same size as `F`.  To
    find `Bd`, we do the following::

            .-      -.                    .-         -.
            |  F  B  |                    |  Phi  Bd  |
        G = |        |    H = expm(G T) = |           |
            |  0  0  |                    |   0   I   |
            '-      -'                    '-         -'

    Note that for `Bd` to be calculated, `B` must have the same number of rows
    as `F`, but need not have the same number of columns.  For `Qd` to be
    calculated, `F` and `Q` must have the same shape.  If these conditions are
    not met, the function will fault.

    We can also express Phi and Bd in terms of their infinite series::

                         1   2  2    1   3  3
        Phi = I + F T + --- F  T  + --- F  T  + ...
                         2!          3!

                    1       2    1   2    3    1   3    4
        Bd = B T + --- F B T  + --- F  B T  + --- F  B T  + ...
                    2!           3!            4!

    The forward Euler method approximations to these are ::

        Phi = I + F T
        Bd  = B T

    The bilinear approximation to Phi is ::

                                         -1/2
        Phi = (I + 0.5 A T) (I - 0.5 A T)

    References
    ----------
    .. [1]  C. Van Loan, "Computing Integrals Involving the Matrix Exponential,"
            1976.
    .. [2]  Brown, R. and Phil Hwang. "Introduction to Random Signals and
            Applied Kalman Filtering (4th ed.)" (2012).
    .. [3]  https://en.wikipedia.org/wiki/Discretization
    """

    # Ensure F, B, and Q are matrices with correct shapes.
    if F is not None:
        if np.ndim(F) == 0:
            F = np.array([[F]])
        elif np.ndim(F) == 1:
            F = np.array([F])
        if F.shape[0] != F.shape[1]:
            raise Exception('vanloan: F must be a square matrix!')
        N = F.shape[1]  # number of states
    else:
        raise Exception('vanloan: F must be provided!')
    if B is not None:
        if np.ndim(B) == 0:
            B = np.array([[B]])
        elif np.ndim(B) == 1:
            B = np.array([B])
        if B.shape[0] != F.shape[0]:
            raise Exception('vanloan: B must have the same number of ' +
                    'rows as F!')
        M = B.shape[1]  # number of inputs
    if Q is not None:
        if np.ndim(Q) == 0:
            Q = np.array([[Q]])
        elif np.ndim(Q) == 1:
            Q = np.array([Q])
        if Q.shape != F.shape:
            raise Exception('vanloan: Q must have the same shape as F!')

    # Get Phi.
    Phi = expm(F*T_Sa)

    # Get Bd.
    if B is not None:
        G = np.vstack(( np.hstack((F, B)), np.zeros((M, N + M)) ))
        H = expm(G*T_Sa)
        Bd = H[0:N, N:(N + M)]
    else:
        Bd = None

    # Get Qd.
    if Q is not None:
        G = np.vstack((
                np.hstack((-F, Q)),
                np.hstack(( np.zeros((N, N)), F.T)) ))
        H = expm(G*T_Sa)
        Qd = Phi.dot(H[0:N, N:(2*N)])
    else:
        Qd = None

    return Phi, Bd, Qd


def xcorr(x):
    """
    Calculate an estimate of the autocorrelation of `x` using a single
    realization.

    Parameters
    ----------
    x : 1D np.ndarray
    C : 1D np.ndarray

    Notes
    -----
    If the length of x is N, then the length of the return value will be 2N-1.
    The return value is normalized such that the middle value will be 1.
    """

    N = len(x)
    C = np.zeros(2*N - 1)
    for n in range(N):
        C[n] = np.sum(x[0:(n + 1)]*x[(N - 1 - n):N])
    C[0:N] /= C[N - 1]
    C[N:] = np.flip(C[0:(N - 1)])

    return C


def avar(y, max_chunks=128):
    """
    Get the Allan variance, `va`, from the noise input
    signal, `y`.

    Notes
    -----
    Note that `y` is only the noise, not the signal.  The Allan variance is
    based on the means of chunks of the noise signal `y`.  We will call the
    chunk size (i.e., the number of samples in a chunk) `m`.  The Allan variance
    is equal to ::

                    K - 1
                    .----
                 1   \         2                       _        _
        va(m) = ---   }   delta               delta  = y      - y
                2 K  /         k                   k    k + m    k
                    '----
                    k = 0                     K = N - 2 m + 1 ,

    where `K` is the number of deltas, `k` is the delta index, and `y hat` is
    the mean of the `m` samples of `y` in a given chunk and its subscript is the
    index of the first value of `y` in the chunk.  Here is an example.  Suppose
    the array `y` has a length of 10::

        y = {y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]} .

    If `m` were 3 and `k` were 0, then ::

        _                               _
        y  = (y[0] + y[1] + y[2])/3     y  = (y[3] + y[4] + y[5])/3 .
         0                               3

    When `k` is 1, we would have ::

        _                               _
        y  = (y[1] + y[2] + y[3])/3     y  = (y[4] + y[5] + y[6])/3 .
         1                               4

    This continues until we reach `k = K - 1`, or `k = N - 2 m`, which in our
    case would be 4::

        _                               _
        y  = (y[4] + y[5] + y[6])/3     y  = (y[7] + y[8] + y[9])/3 .
         4                               7

    Note, we must stop at 4 because this makes the last value we reference
    element `y[9]`, which is the last value of `y`.  So, referencing the
    original equation above, the Allan variance for a given chunk size, `m`, is
    one half of the mean of the square of the difference of means of `y`.  This
    is a mouthful.  But, as you can see, there are a lot of means that must be
    found.  Rather than calculate each mean of `y` when we need it, a more
    efficient method relies on the cumulative sum of the entire input `y` array.
    Here we will show how the cumulative sum array `Y` is used to get the
    deltas.  For the example array `y`, we would have ::

        Y = {y[0], y[0] + y[1], y[0] + y[1] + y[2], ... , y[0] + ... + y[9]} .

    For a `k` value of 1 and an `m` of 3, a simple approach to calculating ::

                 _        _
        delta  = y      - y
             k    k + m    k

    would be ::

        delta[1] = (Y[6] - Y[3])/3 - (Y[3] - Y[0])/3 .

    This works for all values of `k` up to `K-1`.  However, when `k = 0`, we
    find a problem::

        delta[0] = (Y[5] - Y[2])/3 - (Y[2] - Y[-1])/3 .

    There's no such thing as `Y[-1]`.  However, since we know that `Y[j] =
    Y[j-1] + y[j]`, we can say that `Y[-1] = Y[0] - y[0]`::

        delta[0] = (Y[5] - Y[2])/3 - (Y[2] - Y[0] + y[0])/3 .

    Generalizing this, we get ::

        delta[0] = (Y[2*m-1] - 2*Y[m-1] + Y[0] - y[0])/3
        delta[1] = (Y[2*m] - 2*Y[m] + Y[1] - y[1])/3
                ...
        delta[4] = (Y[N-1] - 2*Y[N-1-m] + Y[N-2*m] - y[N-2*m])/3 .

    The above can be vectorized to ::

        delta = (Y[(2*m-1):] - 2*Y[(m-1):(-m)] + Y[:(1-2*m)] - y[:(1-2*m)])/3 .

    Finally, this array of delta values is squared, averaged, and divided by 2.
    The array of chunk sizes, `M`, is related to the array of averaging periods,
    tau, by ::

        tau = M*T .

    The `max_chunks` parameter lets you control the maximum number of chunks you
    will have.  Note, the number of chunks you will actually get will be less
    than this number.

    References
    ----------
    .. [1]  IEEE Std 952-1997
    .. [2]  D. A. Howe, D. W. Allan, J. A. Barnes: "Properties of signal sources
            and measurement methods", pages 464-469, Frequency Control Symposium
            35, 1981.
    .. [3]  https://www.mathworks.com/help/nav/ug/
            inertial-sensor-noise-analysis-using-allan-variance.html
    """

    if len(y.shape) != 1:
        raise Exception('avar: y must be a one-dimensional array.')

    m_max = np.floor((len(y) - 1)/2) if (len(y) > 4) else 2
    M_real = np.logspace(0, np.log10(m_max), max_chunks)
    M = np.unique(np.round(M_real)).astype(int)
    Y = np.cumsum(y)
    va = np.zeros(len(M))
    for n_tau, m in enumerate(M):
        delta = (Y[(2*m - 1):] - 2*Y[(m - 1):(-m)] +
                Y[:(1 - 2*m)] - y[:(1 - 2*m)])/m
        va[n_tau] = np.mean(delta**2)/2

    return va, M


def avar_chunks(N, max_chunks=128):
    """
    Get the array of chunk sizes for the Allan variance function.

    Parameters
    ----------
    N : int
        Number of samples of the noise array.
    max_chunks : int, default 128
        Maximum number of chunks.

    Returns
    -------
    M : 1D np.ndarray
        The array of chunk sizes.
    """

    m_max = np.floor((N - 1)/2) if (N > 4) else 2
    M_real = np.logspace(0, np.log10(m_max), max_chunks)
    M = np.unique(np.round(M_real)).astype(int)
    return M


def armav(va, tau, log_scale=False):
    """
    Solve for five common component noise variances to fit the given total Allan
    variance, `va`, as a function of averaging time, `tau`.

    Parameters
    ----------
    va : 1D np.ndarray
        Total Allan variance array.
    tau : 1D np.ndarray
        Averaging time array.
    log_scale : bool, default False
        Flag to use logarithmic scaling when fitting the component noise
        variances.

    Returns
    -------
    vk : 1D np.ndarray
        Vector of component noise variances.

    Notes
    -----
    The component noise variances are

        - quantization
        - random walk
        - bias instability
        - rate random walk
        - rate ramp

    This algorithm uses the `leastsq` function to solve for the coefficients `k`
    in the equation ::

        y = H k ,

    where `k` is the vector of component noise variances, `y` is the array of
    `N` Allan variances over time, and ::

            .-                                                  -.
            | 3/tau_1^2  1/tau_1  2 ln(2)/pi  tau_1/3  tau_1^2/2 |
        H = |    ...       ...       ...        ...       ...    | .
            | 3/tau_N^2  1/tau_N  2 ln(2)/pi  tau_N/3  tau_N^2/2 |
            '-                                                  -'

    Note that `ln` is the ISO standard notation for the logarithm base e
    (ISO/IEC 80000).

    The third input parameter `log_scale` is a boolean flag that controls
    whether fitting should be done in the log scale base 10 scale or in the
    linear scale.

    References
    ----------
    .. [1]  IEEE Std 952-1997
    .. [2]  Jurado, Juan & Kabban, Christine & Raquet, John. (2019).  A
            regression-based methodology to improve estimation of inertial
            sensor errors using Allan variance data.  Navigation. 66.
            10.1002/navi.278.
    """

    # Ensure all inputs are 1D arrays.
    if len(va.shape) > 1:
        va = va.flatten()
    if len(tau.shape) > 1:
        tau = tau.flatten()

    # linear regression matrix
    H = np.vstack([
        3/tau**2,
        1/tau,
        2*np.log(2)/np.pi + 0*tau,
        tau/3,
        tau**2/2]).T

    # fitting functions
    def flog(k, H, y):
        return np.log10(H.dot(np.absolute(k))) - y

    def flin(k, H, y):
        return H.dot(np.absolute(k)) - y

    # least squares solver
    if log_scale:
        vl = np.log10(va)
        vk, _ = leastsq(flog, np.ones(5), args=(H, vl), maxfev=1000)
        vk = np.absolute(vk)
    else:
        vk, _ = leastsq(flin, np.ones(5), args=(H, va), maxfev=1000)
        vk = np.absolute(vk)

    return vk


def allan_noise(Nt, T_Sa, vk, tau_b=None):
    """
    Build a noise signal of length `Nt`, with time step `T_Sa`.

    Parameters
    ----------
    Nt : int
        Number of samples in the noise array.
    T_Sa : float
        Sampling period in seconds.
    vk : 1D np.ndarray
        Vector of five component noise variances.
    tau_b : float, default None
        Time constant for the bias instability (the first-order Gauss Markov
        noise).

    Returns
    -------
    y : 1D np.ndarray
        The total Allan noise array.

    Notes
    -----
    This noise signal is based on the Allan variances:

        - quantization          vk[0]
        - random walk           vk[1]
        - bias instability      vk[2] and tau_b
        - rate random walk      vk[3]
        - rate ramp             vk[4]

    The rate ramp could be any constant ramp selected from a normal distribution
    with a variance of `vk[4]`.  If any component of `vk` is zero, that noise
    will not be included in the total noise.  If `tau_b` is not supplied, this
    function will guess a good value for it by choosing one half of the value of
    `tau` at the minimum of the ideal Allan variance curve.

    References
    ----------
    .. [1]  IEEE Std 952-1997
    """

    # initial total noise
    y = 0

    # quantization noise
    if vk[0] != 0:
        U = np.random.uniform(-1, 1, Nt + 1)
        y += np.sqrt(3*vk[0])/T_Sa*np.diff(U)

    # random walk noise
    if vk[1] != 0:
        N = np.random.randn(Nt)
        y += np.sqrt(vk[1]/T_Sa)*N

    # bias instability noise
    if vk[2] != 0:
        if tau_b is None:
            tau = avar_chunks(Nt)*T_Sa
            H = np.column_stack((
                3/(tau**2),
                1/tau,
                (2*np.log(2))/np.pi + 0*tau,
                tau/3,
                (tau**2)/2))
            va_ideal = H.dot(vk)
            tau_b = tau[np.argmin(va_ideal)]/2
        nb_t = np.zeros(Nt) # results array
        nb = np.sqrt(vk[2])*np.random.randn(1) # state
        N = np.random.randn(Nt)
        nb_in = np.sqrt(2*vk[2]*T_Sa/tau_b)*N  # input white noise
        for n in range(Nt):
            nb_t[n] = nb
            nb += -(T_Sa/tau_b)*nb + nb_in[n]
        y += nb_t

    # rate random walk noise
    if vk[3] != 0:
        N = np.random.randn(Nt)
        y += np.cumsum(np.sqrt(vk[3]*T_Sa)*N)

    # rate ramp noise
    if vk[4] != 0:
        r0 = np.sqrt(vk[4])*np.random.randn(1)
        y += np.cumsum(r0*T_Sa*np.ones(Nt))

    return y


def sysresample(w):
    """
    Apply systematic resampling based on particle weights.

    Parameters
    ----------
    w : float array_like
        Array of weights, values from 0 to 1.

    Returns
    -------
    ii : int array_like
        Array of new indices after resampling.

    Notes
    -----
    Smaller weights have a lower probability of their corresponding indices
    being selected for the new array ii of sample indices.  This method of
    resampling in comparison with multinomial resampling, stratified resampling,
    and residual resampling was found to offer better resampling quality and
    lower computational complexity [1], [2].  An earlier reference to this
    method appears in [3].

    The cumulative sum, `W`, of the weights, `w`, is created.  Then a
    uniformly-spaced array, `u`, from 0 to `(J-1)/J` is created, where `J` is
    the number of weights.  To this array is added a single
    uniformly-distributed random number in the range `[0, 1/J)`.  Within a loop
    over index `j`, the first value in the array `W` to exceed `u[j]` is noted.
    The index of that value of `W` becomes the `j`th value in an array of
    indices `ii`.  In other words, the first weight to be responsible for
    pushing the cumulative sum of weights past a linearly growing value is
    likely a larger weight and should be reused. ::

        1 -|                            ..........----------````````````
           |                           /
           |                         /
           |                       /
           |                     /
           |         ........../
           |---``````
        0 -|---------+---------+---------+---------+---------+---------+
           0         1         2         3         4         5         6
                               ^
                               This weight contributed greatly to the cumulative
                               sum and is most likely to be selected at this
                               point.

    References
    ----------
    .. [1]  Jeroen D. Hol, Thomas B. Schon, Fredrik Gustafsson, "On Resampling
            Algorithms For Particle Filters," presentated at the Nonlinear
            Statistical Signal Processing Workshop, 2006 IEEE. [Online].
            Available:
            http://users.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
    .. [2]  M. Sanjeev Arulampalam, Simon Maskell, Neil Gordon, and Tim Clapp,
            "A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian
            Bayesian Tracking," IEEE Transactions On Signal Processing, Vol. 50,
            No.  2, Feb. 2002. [Online].  Available:
            https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/
    .. [3]  Genshiro Kitagawa, "Monte Carlo Filter and Smoother for Non-Gaussian
            Nonlinear State Space Models," Journal of Computational and
            Graphical Statistics , Mar., 1996, Vol. 5, No. 1 (Mar., 1996), pp.
            1-25. [Online].  Available:
            https://www.jstor.org/stable/1390750?seq = 1
    """

    J = len(w) # number of particles
    W = np.cumsum(w) # integral of w
    u = (np.arange(J) + np.random.uniform())/J
    i = 0
    ii = np.zeros(J, dtype=int)
    for j in range(J):
        while W[i] < u[j]:
            i += 1
        ii[j] = i
    return ii

# --------
# Plotting
# --------

def plotdensity(t, y, label=None, color='blue', fold=100):
    """
    Create a probability-density contour plot of `y` as a function of `t`.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    y : 1D or 2D np.ndarray
        Abscissa array.
    label : string, default None
        Label string for the legend.
    color : string, default 'blue'
        Color string.
    fold : int, default 100
        The number of times to fold a 1D y array.

    Returns
    -------
    Y : 2D np.ndarray
        Bands of plot.

    Notes
    -----
    The `color` input is the color specification.  For each point along `t`, a
    histogram of all values of the corresponding row of `y` is calculated.  The
    counts of the histogram are compared to 0%, 1.1%, 13.5%, and 60.5% of the
    maximum bin count.  The last three percentages were chosen to correspond to
    3, 2, and 1 standard deviations away from the mean for a normal probability
    distribution.  The indices of the first and last bins to exceed those
    percentage points form the lower and upper edges of each contour. (Actually,
    interpolation is used.)  The point is to show with darker shades where the
    higher density of `y` values are found.  This function does not properly
    handle multi-modal densities.

    `t` must be a single-dimensional array.  `y` must be a matrix where the
    number of rows equals the length of `t`.  However, if `y` is a 1D array,
    then it will be reshaped to a matrix with columns equal to fold.
    """

    # Check the dimensions of the inputs.
    if np.ndim(t) != 1:
        t_shape = t.shape
        if t_shape[0] > 1 and t_shape[1] > 1:
            raise Exception('plotdensity: t must be a vector!')
    if np.ndim(y) == 1:
        # Extend y with mean of ending of y and reshape to a matrix.
        n_last_fold = len(y) - len(y) % fold
        Y_last = np.mean(y[n_last_fold:])
        y_last = Y_last*np.ones(fold - len(y) % fold)
        y = np.append(y, y_last).reshape((-1, fold))
        t = t[::fold]
    elif np.ndim(y) != 2:
        raise Exception('plotdensity: y must be a matrix!')
    y_shape = y.shape
    if y_shape[0] != len(t):
        raise Exception('plotdensity: The length of t must equal to rows of y!')

    # Get the number of row and columns of y.
    rows = y_shape[0]
    cols = y_shape[1]

    # Choose the number of bins and bands.
    bands = 4
    bins = np.ceil(np.sqrt(cols)).astype(int)
    band_heights = np.array([0, 0.011, 0.135, 0.605])

    # Sort the data and add a small amount of spread.  A spread is necessary for
    # the histogram to work correctly.
    y_range = y.max() - y.min()
    dy = y_range*1e-9*(np.arange(cols) - (cols - 1)/2)
    y = np.sort(y, axis=1) + dy

    # Initialize the lower and upper edges of the bands.
    Y = np.zeros((rows, 2*bands))

    # For each row of y,
    for n_row in range(rows):
        # Get this row of y.
        y_row = y[n_row, :]

        # Get the histogram of this row of the y data.
        (h, b) = np.histogram(y_row, bins)

        # Get the mid-points of the bins.
        b = (b[0:bins] + b[1:(bins + 1)])/2

        # Pad the histogram with zero bins.
        db = (b[1] - b[0])*0.5
        b = np.hstack((b[0] - db, b, b[-1] + db))
        h = np.hstack((0, h, 0))

        # Normalize the bin counts.
        h_max = h.max()
        h = h/h_max

        # For this row of y, define the lower and upper edges of the bands.
        Y[n_row, 0] = b[0]
        Y[n_row, 1] = b[-1]
        for n_band in range(1, bands):
            # Get the index before the first value greater than the threshold
            # and the last index of the last value greater than the threshold.
            z = h - band_heights[n_band]
            n = np.nonzero(z >= 0)[0]
            n_a = n[0] - 1
            n_b = n[-1]

            # Interpolate bin locations to find the correct y values of the
            # bands.
            b_a = b[n_a] + (b[n_a + 1] - b[n_a])*(0 - z[n_a])/\
                    (z[n_a + 1] - z[n_a])
            b_b = b[n_b] + (b[n_b + 1] - b[n_b])*(0 - z[n_b])/\
                    (z[n_b + 1] - z[n_b])

            # Store the interpolated bin values.
            Y[n_row, (n_band*2)] = b_a
            Y[n_row, (n_band*2 + 1)] = b_b

    # Plot each band as an overlapping, filled area with 20% opacity.
    plt.fill_between(t.flatten(), Y[:, 0], Y[:, 1],
            alpha=0.2, color=color, label=label)
    for n_band in range(1, bands):
        n_col = (n_band*2)
        plt.fill_between(t.flatten(), Y[:, n_col], Y[:, n_col + 1],
                alpha=0.2, color=color, linewidth=0.0)

    return Y


def plotspan(t, y, label=None, color='blue', fold=100):
    """
    Create an area plot of `y` as a function of `t` showing the minimum and
    maximum envelopes.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    y : 1D or 2D np.ndarray
        Ordinate array.
    label : string, default None
        Label string for the legend.
    color : string, default 'blue'
        Color string.
    fold : int, default 100
        The number of times to fold a 1D y array.

    Returns
    -------
    Y : 2D np.ndarray
        Bands of plot.

    Notes
    -----
    The `color` input is the color specification.  For each point along `t`, the
    minimum and maximum of all values of the corresponding row of `y` are
    calculated.

    `t` must be a single-dimensional array.  `y` must be a matrix where the
    number of rows equals the length of `t`.  However, if `y` is a 1D array,
    then it will be reshaped to a matrix with columns equal to `fold`.
    """

    # Check the dimensions of the inputs.
    if np.ndim(t) != 1:
        t_shape = t.shape
        if t_shape[0] > 1 and t_shape[1] > 1:
            raise Exception('plotdensity: t must be a vector!')
    if np.ndim(y) == 1:
        # Extend y with mean of ending of y and reshape to a matrix.
        n_last_fold = len(y) - len(y) % fold
        Y_last = np.mean(y[n_last_fold:])
        y_last = Y_last*np.ones(fold - len(y) % fold)
        y = np.append(y, y_last).reshape((-1, fold))
        t = t[::fold]
    elif np.ndim(y) != 2:
        raise Exception('plotdensity: y must be a matrix!')
    y_shape = y.shape
    if y_shape[0] != len(t):
        raise Exception('plotdensity: The length of t must equal to rows of y!')

    # Plot band as filled area with 20% opacity.
    Y = np.column_stack((np.min(y, axis=1), np.max(y, axis=1)))
    plt.fill_between(t, Y[:, 0], Y[:, 1],
            alpha = 0.2, color = color, label = label)

    return Y


def hsv_to_rgb(H, S, V):
    """
    Convert hue, saturation, and brightness value to red, green, and blue
    values.

    Parameters
    ----------
    H : {0.0 to 1.0}
        Hue.
    S : {0.0 to 1.0}
        Saturation.
    V : {0.0 to 1.0}
        Brightness value.

    Returns
    -------
    R : {0.0 to 1.0}
        Red.
    G : {0.0 to 1.0}
        Green.
    B : {0.0 to 1.0}
        Blue.
    V : {0.0 to 1.0}
        Actual brightness.
    """

    # Normalize and bound the inputs.
    H -= np.round(H)
    S = np.clip(S, 0, 1)
    V = np.clip(V, 0, 1)

    # Initialize the red, green, blue values.
    r = 1
    g = 1
    b = 1

    tH = np.tan(H*2*np.pi)
    sq3 = np.sqrt(3)

    if H < -0.333:      # -180 to -120 degrees  max: B, min: R
        r = b*(1 - S)
        g = ((2*r - b)*tH + sq3*b)/(sq3 + tH)
    elif H < -0.167:    # -120 to  -60 degrees  max: B, min: G
        g = b*(1 - S)
        r = ((b + g)*tH + sq3*(g - b))/(2*tH)
    elif H < 0:         #  -60 to    0 degrees  max: R, min: G
        g = r*(1 - S)
        b = ((2*r - g)*tH - sq3*g)/(tH - sq3)
    elif H < 0.167:     #    0 to   60 degrees  max: R, min: B
        b = r*(1 - S)
        g = ((2*r - b)*tH + sq3*b)/(sq3 + tH)
    elif H < 0.333:     #   60 to  120 degrees  max: G, min: B
        b = g*(1 - S)
        r = ((b + g)*tH + sq3*(g - b))/(2*tH)
    else:               #  120 to  180 degrees  max: G, min: R
        r = g*(1 - S)
        b = ((2*r - g)*tH - sq3*g)/(tH - sq3)

    # Correct the brightness value (See https://alienryderflex.com/hsp.html).
    v = np.sqrt(0.299*r*r + 0.587*g*g + 0.114*b*b)
    k = V/v
    if k > 1:
        k = 1.0
        V = v
    R = r*k
    G = g*k
    B = b*k

    # Round to 8-bit precision.
    R = np.round(R*255)/255
    G = np.round(G*255)/255
    B = np.round(B*255)/255

    # Return the red, green, blue, and actual brightness values.
    return R, G, B, V


def rgb_to_hsv(R, G, B):
    """
    Convert red, green, and blue values to hue, saturation, and brightness
    value.

    Parameters
    ----------
    R : {0.0 to 1.0}
        Red.
    G : {0.0 to 1.0}
        Green.
    B : {0.0 to 1.0}
        Blue.

    Returns
    -------
    H : {0.0 to 1.0}
        Hue.
    S : {0.0 to 1.0}
        Saturation.
    V : {0.0 to 1.0}
        Brightness value.
    """

    # Bound the inputs.
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)

    # Get the saturation.
    M = max(R, G, B)
    if M > 0:
        S = (M - min(R, G, B))/M
    else:
        S = 0

    # Get the hue.
    H = np.arctan2(np.sqrt(3)*(G - B), 2*R - G - B)/(2*np.pi)

    # Get the brightness value.
    V = np.sqrt(0.299*R*R + 0.587*G*G + 0.114*B*B)

    return H, S, V


def hex_to_rgb(color):
    """
    Convert 24-bit color values to red, green, blue values on the scale of 0
    to 255.  This function can take an ndarray of colors.
    """

    B = np.bitwise_and(color, 0xFF)
    RG = np.right_shift(color, 8)
    G = np.bitwise_and(RG, 0xFF)
    R = np.right_shift(RG, 8)
    return R, G, B


def rgb_to_hex(R, G, B):
    """
    Convert red, green, blue color values on the scale of 0 to 255 to 24-bit
    color values.  This function can take ndarrays of `R`, `G`, and `B`.
    """

    color = np.left_shift(R, 16) + np.left_shift(G, 8) + B
    return color


def mix_colors(C0, C1, w):
    """
    Mix colors `C0` and `C1` using weight `w`.  A weight of 0 makes the
    output equal to `C0`, and a weight of 1 makes the output equal to `C1`.
    The colors are treated as 6-character hexadecimal values.
    """

    R0, G0, B0 = hex_to_rgb(C0)
    R1, G1, B1 = hex_to_rgb(C1)
    R = np.sqrt(w*R1**2 + (1 - w)*R0**2).astype(int)
    G = np.sqrt(w*G1**2 + (1 - w)*G0**2).astype(int)
    B = np.sqrt(w*B1**2 + (1 - w)*B0**2).astype(int)
    color = rgb_to_hex(R, G, B)
    return color

# ---------
# Animation
# ---------

class shape:
    """
    Animation shape class.

    See Also
    --------
    frame
    animate
    """

    def __init__(self, x = 0.0, y = 0.0, stroke = 1.0, color = 0x000,
            alpha = 1.0, T_Sa = 1.0):
        """
        Initialize the shape object.

        Parameters
        ----------
        x : float or array_like, default 0.0
            x values of the shape or the x-axis radius of an ellipse.
        y : float or array_like, default 0.0
            y values of the shape or the y-axis radius of an ellipse.
        stroke : float or array_like, default 1.0
            Stroke width or animated widths.
        color : float or array_like, default 0x000
            Stroke or fill color or animated colors.  It is a fill color or
            colors if `stroke` is a scalar zero.
        alpha : float or array_like, default 1.0
            Opacity or animated opacity values.  0.0 means fully transparent and
            1.0 means fully opaque.
        T_Sa : float, default 1.0
            Sampling period in seconds.
        """

        def to_ints(x):
            """
            Convert x to an integer or array of integers.
            """

            if np.ndim(x) == 0:
                x = int(x)
            if isinstance(x, list):
                x = np.array(x, dtype=int)
            return x

        def to_floats(x):
            """
            Convert x to a float or array of floats.
            """

            if (np.ndim(x) == 0):
                x = float(x)
            if isinstance(x, list):
                x = np.array(x, dtype=float)
            return x

        # Ensure x, y, stroke, and alpha are either floats or ndarrays.  Ensure
        # color is either an int or an ndarray.
        x = to_floats(x)
        y = to_floats(y)
        stroke = to_floats(stroke)
        color = to_ints(color)
        alpha = to_floats(alpha)

        # Ensure if either x or y is an ndarray, they both are.
        if isinstance(x, float) and isinstance(y, np.ndarray):
            x = x*np.ones(len(y))
        elif isinstance(x, np.ndarray) and isinstance(y, float):
            y = y*np.ones(len(x))

        # Ensure x and y are the same length.
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if len(x) != len(y):
                raise Exception('shape: x and y must be the same length!')

        # Ensure stroke, color, and alpha are non-negative and that alpha is not
        # more than one.
        if isinstance(stroke, float):
            if stroke < 0.0:
                stroke = 0.0
        else:
            stroke[stroke < 0.0] = 0.0
        if isinstance(color, int):
            if color < 0:
                color = 0
            elif color > 0xffffff:
                color = 0xffffff
        else:
            color[color < 0] = 0
            color[color > 0xffffff] = 0xffffff
        if isinstance(alpha, float):
            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0
        else:
            alpha[alpha < 0.0] = 0.0
            alpha[alpha > 1.0] = 1.0

        # If stroke, color, or alpha are arrays and T_Sa is None, default to 1.
        if (isinstance(stroke, np.ndarray) or isinstance(color, np.ndarray) or \
                isinstance(alpha, np.ndarray)) and (T_Sa is None):
            T_Sa = 1

        # Add these parameters to the object.
        self.x = x
        self.y = y
        self.stroke = stroke
        self.color = color
        self.alpha = alpha
        self.T_Sa = T_Sa


class frame:
    """
    Animation shape class.

    See Also
    --------
    shape
    animate
    """

    def __init__(self, objs, x = 0.0, y = 0.0, ang = 0.0, scale = 1.0,
            T_Sa = 1.0):
        """
        Initialize the frame object.

        Parameters
        ----------
        objs : shape or frame or list of such
            A single shape or frame object or a list of such objects.
        x : float or array_like, default 0.0
            x-axis values of translation of the objects within the frame.
        y : float or array_like, default 0.0
            y-axis values of translation of the objects within the frame.
        ang : float or array_like, default 0.0
            Angles of rotation of the objects within the frame in radians.
        scale : float or array_like, default 1.0
            Scaling factors of the objects within the frame.
        T_Sa : float, default 1.0
            Sampling period in seconds.
        """

        # Ensure objs is a list.
        if isinstance(objs, (shape, frame)):
            objs = [objs]

        def to_floats(x):
            """
            Convert x to a float or array of floats.
            """

            if np.ndim(x) == 0:
                x = float(x)
            if isinstance(x, list):
                x = np.array(x, dtype=float)
            return x

        # Ensure x, y, ang, and scale are either floats or ndarrays.
        x = to_floats(x)
        y = to_floats(y)
        ang = to_floats(ang)
        scale = to_floats(scale)

        # Ensure if either x or y is an ndarray, they both are.
        if isinstance(x, float) and isinstance(y, np.ndarray):
            x = x*np.ones(len(y))
        elif isinstance(x, np.ndarray) and isinstance(y, float):
            y = y*np.ones(len(x))
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if len(x) != len(y):
                raise Exception('frame: x and y must be the same length!')

        # Ensure scale is non-negative.
        if isinstance(scale, float):
            if scale < 0.0:
                scale = 0.0
        else:
            scale[scale < 0.0] = 0.0

        # If x, y, ang, or scale are arrays and T_Sa is None, default to 1.
        if (isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or
                isinstance(ang, np.ndarray)) and (T_Sa is None):
            T_Sa = 1

        # Add these parameters to the object.
        self.objs = objs
        self.x = x
        self.y = y
        self.ang = ang
        self.scale = scale
        self.T_Sa = T_Sa


def jet_frames(x = 0.0, y = 0.0, ang = 0.0, force = 1.0,
        C0 = 0xF67926, C1 = 0xFFFBCE,
        width = 1.0, ratio = 4.0, cnt = 40, T_Sa = 0.05):
    """
    Create a list of animation frames, each one of a moving, fading bubble
    (circle), designed to imitate a jet of gas or fire.

    Parameters
    ----------
    x : float or array_like, default 0.0
        x position over time of jet source.
    y : float or array_like, default 0.0
        y position over time of jet source.
    ang : float or array_like, default 0.0
        Angle relative to x axis over time of jet source.  This is the opposite
        of the direction the jet bubbles move.
    C0 : int, default 0xF67926
        24-bit RGB color of bottom layer of bubbles.  The default is a dark
        orange.
    C1 : int, default 0xFFFBCE
        24-bit RGB color of top layer of bubbles.  The default is a bright
        yellow.
    width : float, default 1.0
        Width of jet.
    ratio : float, default 2.0
        Ratio of length to width of jet.
    cnt : int, default 50
        Number of bubbles to draw.
    T_Sa : float, default 0.05
        Sampling period in seconds.

    Returns
    -------
    obj_list : list of frame objects
        List of frame objects, each with one bubble (circle) shape.

    Notes
    -----
    The jet is by default pointed to the right (the bubbles move to the left),
    with the center of the starting point of the largest bubbles centered at
    `(0, 0)`.

    The bubbles are each within their own frame which loops about every second.
    The `x`, `y`, and `ang` arrays are used to then position and rotate those
    frames but only between the end and beginning of a bubble's loop.  The total
    duration of the animation is fixed.  Ideally, the duration of the bubble's
    loop would be an integer fraction of the total animation duration and an
    integer multiple of the sampling period.  However, if the total duration is
    a prime number multiple of the sampling period, this ideal is not possible.
    Something must give.  For the bubble animation to align with the total
    animation, the duration of the bubble loop must be an integer fraction of
    the total duration.  Therefore, the duration of the bubble loop cannot be
    guaranteed to be an integer multiple of the sampling period.  This means
    that `x`, `y`, and `ang` must be linearly interpolated over time to get the
    correct values at the beginnings of each bubble loop iteration.
    """

    # Convert x, y, or ang lists to np.ndarrays.
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(ang, list):
        ang = np.array(ang)

    # Ensure x, y, and ang arrays are 1D and get their lengths.
    if isinstance(x, np.ndarray):
        if x.ndim > 1:
            x = x.flatten()
        N_x = len(x)
    else:
        N_x = 1
    if isinstance(y, np.ndarray):
        if y.ndim > 1:
            y = y.flatten()
        N_y = len(y)
    else:
        N_y = 1
    if isinstance(ang, np.ndarray):
        if ang.ndim > 1:
            ang = ang.flatten()
        N_ang = len(ang)
    else:
        N_ang = 1
    if isinstance(force, np.ndarray):
        if force.ndim > 1:
            force = force.flatten()
        N_force = len(force)
    else:
        N_force = 1

    # Ensure x, y, and ang arrays are the same shape.
    xy_diff = (N_x > 1) and (N_y > 1) and (N_x != N_y)
    xa_diff = (N_x > 1) and (N_ang > 1) and (N_x != N_ang)
    ya_diff = (N_y > 1) and (N_ang > 1) and (N_y != N_ang)
    if xy_diff or xa_diff or ya_diff:
        raise Exception('jet_frames: if x, y, or ang are arrays, ' +
                'they must be the same shape.')

    # Get the total duration of the animation.
    N_tot = max(N_x, N_y, N_ang)
    if N_tot == 1:
        T_tot = 5.0 # default duration [s]
        N = round(T_tot/T_Sa)
        T_tot = N*T_Sa
    else:
        T_tot = N_tot*T_Sa
        t_tot = np.arange(N_tot)*T_Sa # array of time for x, y, and ang

    # For each bubble,
    obj_list = []
    for n_bubble in range(cnt):
        # Get the duration of the bubble animation loop as an integer fraction
        # of the total duration.
        T_rep = 0.1 + 0.3*np.random.rand()
        N_reps = round(T_tot/T_rep)
        if N_reps < 1:
            N_reps = 1
        T_rep = T_tot/N_reps

        # Get the animation sampling period as an integer fraction of the
        # bubble animation loop duration.  It does not have to be T_Sa.
        T_ani = 0.01
        N_ani = round(T_rep/T_ani)
        if N_ani < 10:
            N_ani = 10
        T_ani = T_rep/N_ani

        # Get scaling factor.
        r = float(n_bubble)/(cnt - 1) # progress through the animation [0, 1]
        scale = np.linspace(1, 1.5, N_ani)*(1.0 - 0.5*r) # [1, 1.5], [0.5, 0.75]

        # Get the bubble x and y path values.
        length = width*ratio
        x_begin = 0.5*width*(1.0 - scale[0])
        x_end = -length*2*(1 + np.random.rand())/3
        x_bubble = np.linspace(x_begin, x_end, N_ani)
        y_end = (0.5*width)*np.random.randn()
        y_end -= np.round(y_end/width)*width
        y_bubble = np.linspace(0, y_end, N_ani)

        # Get alpha.
        alpha = np.linspace(1, 0, N_ani)
        alpha[0] = 0

        # Get color by averaging.
        color = mix_colors(C0, C1, r)

        # Define bubble (circle).
        curve = shape(0.5*width, 0.5*width, stroke = 0, color = color,
                alpha = alpha, T_Sa = T_ani)

        # Define bubble frame.
        bubble = frame(curve, x_bubble, y_bubble, scale=scale, T_Sa=T_ani)

        # Move and rotate the bubble frame for dynamic x, y, and ang.
        if N_tot > 1:
            # Get full animation time array.
            t_ani = np.arange(N_ani*N_reps)*T_ani

            # Interpolate to get x, y, and ang values at starts of bubble loops.
            t_reps = np.arange(N_reps)*T_rep
            if (N_x > 1):
                x_reps = np.interp(t_reps, t_tot, x)
                x_ani = np.zeros(len(t_ani))
                for n_rep in range(N_reps):
                    na = n_rep*N_ani
                    x_ani[na:(na + N_ani)] = x_reps[n_rep]
            else:
                x_ani = x
            if (N_y > 1):
                y_reps = np.interp(t_reps, t_tot, y)
                y_ani = np.zeros(len(t_ani))
                for n_rep in range(N_reps):
                    na = n_rep*N_ani
                    y_ani[na:(na + N_ani)] = y_reps[n_rep]
            else:
                y_ani = y
            if (N_ang > 1):
                ang_reps = np.interp(t_reps, t_tot, ang)
                ang_ani = np.zeros(len(t_ani))
                for n_rep in range(N_reps):
                    na = n_rep*N_ani
                    ang_ani[na:(na + N_ani)] = ang_reps[n_rep]
            else:
                ang_ani = ang
            if (N_force > 1):
                force_reps = np.interp(t_reps, t_tot, force)
                force_ani = np.zeros(len(t_ani))
                for n_rep in range(N_reps):
                    na = n_rep*N_ani
                    force_ani[na:(na + N_ani)] = force_reps[n_rep]
            else:
                force_ani = force

            # Put the bubble into another frame.
            bubble = frame(bubble, x_ani, y_ani, ang = ang_ani,
                    scale = force, T_Sa = T_ani)

        # Append to the list of cnt.
        obj_list.append(bubble)

    # Move and rotate the bubble frames for static x, y, and ang.
    if N_tot == 1:
        obj_list = frame(obj_list, x, y, ang)

    return obj_list


def animate(obj, filename='ani.svg', x_lim=[-1, 1], y_lim=[-1, 1]):
    """
    Create an svg (scalable vector graphics) animation file from the data stored
    in `obj`.

    Parameters
    ----------
    obj : shape object, frame object, or list of such objects
        The data to create an animation
    filename : string, default 'ani.svg'
        The desired name of output file.  End it with '.svg' in order for
        your system to automatically know how to open it.
    x_lim : array_like, default [-1, 1]
        A list or array of the minimum and maximum x values to display in
        the animation.
    y_lim : array_like, default [-1, 1]
        A list or array of the minimum and maximum y values to display in
        the animation.
    """

    def to_str(x):
        """
        Convert `x` from a float to a string or from an array of floats to a
        list of strings.
        """

        if isinstance(x, (np.ndarray, list)):
            return ['%.5g' % (x[n]) for n in range(len(x))]
        else:
            return '%.5g' % (x)

    def svg_color(C):
        """
        Convert `C` from a color integer to an svg-style color string or from an
        array of integers to a list of strings.
        """

        if isinstance(C, (np.ndarray, list)):
            return ['#' + hex(C[n])[2:].zfill(3) if (C[n] <= 0xfff) else
                    '#' + hex(C[n])[2:].zfill(6) for n in range(len(C))]
        else:
            return ('#' + hex(C)[2:].zfill(3) if (C <= 0xfff) else
                    '#' + hex(C)[2:].zfill(6))

    def keypoints(y, tol=0.0):
        """
        Find the array of indices of `y` where `y` changes slope by more than
        `tol`.  This includes the end points: 0 and N - 1.
        """

        # Get the indices of the starts of keypoints.
        dy = np.diff(y)
        ddy = np.diff(dy)
        n_start = np.nonzero(np.abs(ddy) > tol + 1e-12)[0] + 1

        # Build the full index array.
        nn = np.zeros(len(n_start) + 2, dtype=int)
        nn[1:-1] = n_start
        nn[-1] = len(y) - 1

        return nn

    def add_values(a, b = None, level = 1, pre = '', post = '', sep = '',
            end = ''):
        """
        Write the elements of list `a` (and of list `b` if it is not `None`).

        Parameters
        ----------
        a : string array_like
            First list of strings of values to add.
        b : string array_like, default None
            Second list of strings of values to add.
        level : int, default 1
            Indentation level.
        pre : string, default ''
            String to place before any values.
        post : string, default ''
            String to place after the first or first pair of values.
        sep : string, default ''
            String (besides space) to place between values or pairs of values.
        end : string, default ''
            String  to place after the last value or pair of values.
        """

        ind = '  '*level
        if b is None:
            text = ind + '\"' + pre + a[0] + post + sep + ' '
            for n in range(1, len(a) - 1):
                text += a[n]
                if len(text + sep + a[n + 1]) >= 80:
                    text += sep + '\n'
                    fid.write(text)
                    text = ind
                else:
                    text += sep + ' '
            text += a[-1] + end + '\"'
            fid.write(text)
        else:
            text = ind + '\"' + pre + a[0] + ',' + b[0] + post + sep + ' '
            for n in range(1, len(a) - 1):
                text += a[n] + ',' + b[n]
                if len(text + sep + a[n + 1] + ',' + b[n + 1]) >= 80:
                    text += sep + '\n'
                    fid.write(text)
                    text = ind
                else:
                    text += sep + ' '
            text += a[-1] + ',' + b[-1] + end + '\"'
            fid.write(text)

    def dynamic_translate(obj, scale_post, level):
        """
        Add translation (x or y movement) animation.

        Parameters
        ----------
        obj : frame
            Frame object with `x` and `y` position arrays and `T_Sa` sampling
            period.
        scale_post : float
            Scaling factor that will be applied to the whole frame.
        level : int
            Indentation level.
        """

        # Introduce the animation.
        ind = '  '*level
        fid.write('%s<animateTransform\n' % (ind))
        fid.write('%s  attributeName=\"transform\"\n' % (ind))
        fid.write('%s  type=\"translate\"\n' % (ind))
        fid.write('%s  additive=\"sum\"\n' % (ind))
        fid.write('%s  repeatCount=\"indefinite\"\n' % (ind))
        fid.write('%s  dur=\"%.15gs\"\n' % (ind, len(obj.x)*obj.T_Sa))

        # Get the scaled animation values and key-point indices.
        px =  np.append(obj.x, obj.x[0])*p_scale/scale_post
        py = -np.append(obj.y, obj.y[0])*p_scale/scale_post
        nn_px_keys = keypoints(px)
        nn_py_keys = keypoints(py)
        nn_keys = np.unique(np.concatenate((nn_px_keys, nn_py_keys)))

        # Write the values at key points or write all the values.
        if len(nn_keys) < len(px)*0.5:
            fid.write('%s  keyTimes=\n' % (ind))
            key_times = nn_keys/float(len(px) - 1)
            key_times[-1] = 1
            key_times = to_str(key_times)
            add_values(key_times, None, level + 2, sep=';')
            fid.write('\n%s  values=\n' % (ind))
            px = to_str(px[nn_keys])
            py = to_str(py[nn_keys])
            add_values(px, py, level + 2, sep=';')
        else:
            fid.write('%s  values=\n' % (ind))
            px = to_str(px)
            py = to_str(py)
            add_values(px, py, level + 2, sep=';')
        fid.write('/>\n')

    def dynamic_rotate(obj, level):
        """
        Add rotation animation.

        Parameters
        ----------
        obj : frame
            Frame object with `ang` rotation array and `T_Sa` sampling period.
        level : int
            Indentation level.

        Notes
        -----
        Although this is meant for dynamic rotation only, if there is also
        dynamic translation, a static rotation cannot be performed in the group
        opening instructions because that will cause the rotation to occur after
        the translation.  We always want the rotation to occur before the
        translation.  So, only this dynamic also handles some static cases.
        """

        # Introduce the animation.
        ind = '  '*level
        fid.write('%s<animateTransform\n' % (ind))
        fid.write('%s  attributeName=\"transform\"\n' % (ind))
        fid.write('%s  type=\"rotate\"\n' % (ind))
        fid.write('%s  additive=\"sum\"\n' % (ind))
        fid.write('%s  repeatCount=\"indefinite\"\n' % (ind))
        fid.write('%s  dur=\"%.15gs\"\n' % (ind, len(obj.ang)*obj.T_Sa))

        # Get the scaled animation values and key-point indices.
        ang = -np.unwrap(np.append(obj.ang, obj.ang[0]))*RAD_TO_DEG
        nn_keys = keypoints(ang)

        # Write the values at key points or write all the values.
        if len(ang) == 2: # for static rotation before dynamic translation
            fid.write('%s  values=\"%.3g\"' % (ind, ang[0]))
        elif len(nn_keys) < len(ang)*0.5:
            fid.write('%s  keyTimes=\n' % (ind))
            key_times = nn_keys/float(len(ang) - 1)
            key_times[-1] = 1
            key_times = to_str(key_times)
            add_values(key_times, None, level + 2, sep=';')
            fid.write('\n%s  values=\n' % (ind))
            ang = to_str(ang[nn_keys])
            add_values(ang, None, level + 2, sep=';')
        else:
            fid.write('%s  values=\n' % (ind))
            ang = to_str(ang)
            add_values(ang, None, level + 2, sep=';')
        fid.write('/>\n')

    def dynamic_scale(obj, scale_post, level):
        """
        Add scaling factor animation.

        Parameters
        ----------
        obj : frame
            Frame object with `scale` factor array and `T_Sa` sampling period.
        scale_post : float
            Scaling factor that will be applied to the whole frame.
        level : int
            Indentation level.
        """

        # Introduce the animation.
        ind = '  '*level
        fid.write('%s<animateTransform\n' % (ind))
        fid.write('%s  attributeName=\"transform\"\n' % (ind))
        fid.write('%s  type=\"scale\"\n' % (ind))
        fid.write('%s  additive=\"sum\"\n' % (ind))
        fid.write('%s  repeatCount=\"indefinite\"\n' % (ind))
        fid.write('%s  dur=\"%.15gs\"\n' % (ind, len(obj.scale)*obj.T_Sa))

        # Get the scaled animation values and key-point indices.
        scales = np.append(obj.scale, obj.scale[0])/scale_post
        nn_keys = keypoints(scales)

        # Write the values at key points or write all the values.
        if len(nn_keys) < len(scales)*0.5:
            fid.write('%s  keyTimes=\n' % (ind))
            key_times = nn_keys/float(len(scales) - 1)
            key_times[-1] = 1
            key_times = to_str(key_times)
            add_values(key_times, None, level + 2, sep=';')
            fid.write('\n%s  values=\n' % (ind))
            scales = to_str(scales[nn_keys])
            add_values(scales, None, level + 2, sep=';')
        else:
            fid.write('%s  values=\n' % (ind))
            scales = to_str(scales)
            add_values(scales, None, level + 2, sep=';')
        fid.write('/>\n')

    def dynamic_stroke(obj, level):
        """
        Add stroke width animation.

        Parameters
        ----------
        obj : shape
            Shape object with `stroke` width array and `T_Sa` sampling period.
        level : int
            Indentation level.
        """

        # Introduce the animation.
        ind = '  '*level
        fid.write('%s<animate\n' % (ind))
        fid.write('%s  attributeName=\"stroke-width\"\n' % (ind))
        fid.write('%s  repeatCount=\"indefinite\"\n' % (ind))
        fid.write('%s  dur=\"%.15gs\"\n' % (ind, len(obj.stroke)*obj.T_Sa))

        # Get the scaled animation values and key-point indices.
        strokes = np.append(obj.stroke, obj.stroke[0])*p_scale
        nn_keys = keypoints(strokes)

        # Write the values at key points or write all the values.
        if len(nn_keys) < len(strokes)*0.5:
            fid.write('%s  keyTimes=\n' % (ind))
            key_times = nn_keys/float(len(strokes) - 1)
            key_times[-1] = 1
            key_times = to_str(key_times)
            add_values(key_times, None, level + 2, '', '', ';', '')
            fid.write('\n%s  values=\n' % (ind))
            strokes = to_str(strokes[nn_keys])
            add_values(strokes, None, level + 2, '', '', ';', '')
        else:
            fid.write('%s  values=\n' % (ind))
            strokes = to_str(strokes)
            add_values(strokes, None, level + 2, '', '', ';', '')
        fid.write('/>\n')

    def dynamic_color(obj, level):
        """
        Add stroke or fill color animation.  Which is determined by the `stroke`
        value.

        Parameters
        ----------
        obj : shape
            Shape object with `color` and `stroke` width arrays and `T_Sa`
            sampling period.
        level : int
            Indentation level.
        """

        # Introduce the animation.
        ind = '  '*level
        fid.write('%s<animate\n' % (ind))
        if isinstance(obj.stroke, float) and (obj.stroke <= 0):
            fid.write('%s  attributeName=\"fill\"\n' % (ind))
        else:
            fid.write('%s  attributeName=\"stroke\"\n' % (ind))
        fid.write('%s  repeatCount=\"indefinite\"\n' % (ind))
        fid.write('%s  dur=\"%.15gs\"\n' % (ind, len(obj.color)*obj.T_Sa))

        # Get the scaled animation values and key-point indices.
        colors = np.append(obj.color, obj.color[0])
        nn_keys = keypoints(colors)

        # Write the values at key points or write all the values.
        if len(nn_keys) < len(colors)*0.5:
            fid.write('%s  keyTimes=\n' % (ind))
            key_times = nn_keys/float(len(colors) - 1)
            key_times[-1] = 1
            key_times = to_str(key_times)
            add_values(key_times, None, level + 2, '', '', ';', '')
            fid.write('\n%s  values=\n' % (ind))
            colors = svg_color(colors)
            add_values(colors, None, level + 2, '', '', ';', '')
        else:
            fid.write('%s  values=\n' % (ind))
            colors = svg_color(colors)
            add_values(colors, None, level + 2, '', '', ';', '')
        fid.write('/>\n')

    def dynamic_alpha(obj, level):
        """
        Add alpha (opacity) animation.

        Parameters
        ----------
        obj : shape
            Shape object with `alpha` array and `T_Sa` sampling period.
        level : int
            Indentation level.
        """

        # Introduce the animation.
        ind = '  '*level
        fid.write('%s<animate\n' % (ind))
        fid.write('%s  attributeName=\"opacity\"\n' % (ind))
        fid.write('%s  repeatCount=\"indefinite\"\n' % (ind))
        fid.write('%s  dur=\"%.15gs\"\n' % (ind, len(obj.alpha)*obj.T_Sa))

        # Get the scaled animation values and key-point indices.
        alphas = np.append(obj.alpha, obj.alpha[0])
        nn_keys = keypoints(alphas)

        # Write the values at key points or write all the values.
        if len(nn_keys) < len(alphas)*0.5:
            fid.write('%s  keyTimes=\n' % (ind))
            key_times = nn_keys/float(len(alphas) - 1)
            key_times[-1] = 1
            key_times = to_str(key_times)
            add_values(key_times, None, level + 2, '', '', ';', '')
            fid.write('\n%s  values=\n' % (ind))
            alphas = to_str(alphas[nn_keys])
            add_values(alphas, None, level + 2, '', '', ';', '')
        else:
            fid.write('%s  values=\n' % (ind))
            alphas = to_str(alphas)
            add_values(alphas, None, level + 2, '', '', ';', '')
        fid.write('/>\n')

    def add_shape(obj, level):
        """
        Add shape object details: the shape itself as defined by (`x`,`y`),
        `stroke` width, stroke or fill `color`, and `alpha`.

        Parameters
        ----------
        obj : shape
            Shape object.
        level : int
            Indentation level.

        Notes
        -----
        First, the type of shape needs to be determined.  If (`x`,`y`) is a
        single pair of scalar values, then this is either a circle or and
        ellipse where `x` is the x-axis radius and `y` is the y-axis radius.  If
        `x` equals `y` then it is a circle, and if not, then it is an ellipse.
        If (`x`,`y`) is a pair of vectors, then a generic path is created.

        Second, any static properties of the shape are specified.

        Third, the shape itself is defined.

        Finally, any dynamic properties are specified.
        """

        # Start the shape.
        ind = '  '*level
        if isinstance(obj.x, float):
            if obj.x == obj.y:
                fid.write('%s<circle\n' % (ind))
            else:
                fid.write('%s<ellipse\n' % (ind))
        else:
            fid.write('%s<path\n' % (ind))

        # Write the static stroke, color, and alpha.
        if isinstance(obj.stroke, float) and (obj.stroke <= 0):
            do_fill = True
        else:
            do_fill = False
        if not do_fill:
            fid.write('%s  vector-effect=\"non-scaling-stroke\"\n' % (ind))
            fid.write('%s  fill-opacity=\"0\"\n' % (ind))
            fid.write('%s  stroke-linecap=\"round\"\n' % (ind))
            fid.write('%s  stroke-linejoin=\"round\"\n' % (ind))
        if isinstance(obj.stroke, float):
            fid.write('%s  stroke-width=\"%.3g\"\n' %
                    (ind, obj.stroke*p_scale))
        if isinstance(obj.color, int):
            color_str = svg_color(obj.color)
            if do_fill:
                fid.write('%s  fill=\"%s\"\n' % (ind, color_str))
            else:
                fid.write('%s  stroke=\"%s\"\n' % (ind, color_str))
        if isinstance(obj.alpha, float) and (obj.alpha != 1):
            fid.write('%s  opacity=\"%s\"\n' % (ind, to_str(obj.alpha)))

        # Write the x and y values.
        if isinstance(obj.x, float):
            if (obj.x == obj.y):
                fid.write('%s  r=\"%.3g\"' % (ind, abs(obj.x*p_scale)))
            else:
                fid.write('%s  rx=\"%.3g\"\n' % (ind, abs(obj.x*p_scale)))
                fid.write('%s  ry=\"%.3g\"' % (ind, abs(obj.y*p_scale)))
        else:
            fid.write('%s  d=\n' % (ind))
            px = to_str( obj.x*p_scale)
            py = to_str(-obj.y*p_scale)
            if do_fill:
                add_values(px, py, level + 2, 'M ', ' L', '', ' z')
            else:
                add_values(px, py, level + 2, 'M ', ' L', '', '')

        # Write the ending with any dynamics.
        if isinstance(obj.stroke, float) and isinstance(obj.color, int) and \
                isinstance(obj.alpha, float):
            fid.write('/>\n')
        else:
            fid.write('>\n')
            if isinstance(obj.stroke, np.ndarray):
                dynamic_stroke(obj, level + 1)
            if isinstance(obj.color, np.ndarray):
                dynamic_color(obj, level + 1)
            if isinstance(obj.alpha, np.ndarray):
                dynamic_alpha(obj, level + 1)
            if not isinstance(obj.x, np.ndarray):
                if obj.x == obj.y:
                    fid.write('%s</circle>\n' % (ind))
                else:
                    fid.write('%s</ellipse>\n' % (ind))
            else:
                fid.write('%s</path>\n' % (ind))

    def add_obj(obj, level):
        """
        Add the shape, frame, or list of such to the svg file.

        Parameters
        ----------
        obj : shape, frame, or list of such
            A shape or frame object or a list of such objects to be added to the
            svg file.
        level : int
            Indentation level.

        Notes
        -----
        If the object is a frame, create an svg group, specify any static
        translate, rotate, or scale properties, add the child objects, add any
        dynamic (animated) translate, rotate, or scale properties, and close the
        group.  To avoid visual distortions visible in some web browsers due to
        animated down-scaling, get the minimum animated scaling factor, divide
        all the animated scaling factor values by this minimum so that the new
        minimum is 1, and post scale down the whole group.  The `scale_post` is
        this minimum scaling factor.

        If the object is a shape, add the shape.

        If the object is a list, call this function on each of the members.
        """

        # Get the indent string.
        ind = '  '*level

        if isinstance(obj, frame):
            # Open the group.
            fid.write('%s<g' % (ind))

            # Get static flags.
            xy_static = isinstance(obj.x, (float, int))
            ang_static = isinstance(obj.ang, (float, int))
            scale_static = isinstance(obj.scale, (float, int))

            # Get the post scaling factor.
            if not scale_static:
                scale_post = np.min(obj.scale)
                if scale_post < 1e-3:
                    scale_post = 1e-3
                if abs(scale_post - 1) < 1e-3:
                    scale_post = 1.0
            else:
                scale_post = 1.0

            # Write any static settings.  This includes static down-scaling
            # after dynamic up-scaling (to overcome artifacts in Safari).
            if (xy_static and ((obj.x != 0) or (obj.y != 0))) or \
                    (xy_static and ang_static) or \
                    (scale_static and (obj.scale != 1)) or (scale_post != 1):
                fid.write(' transform=\"')
                is_first = True
                if xy_static and ((obj.x != 0) or (obj.y != 0)):
                    fid.write('translate(%.3f,%.3f)' %
                            (obj.x*p_scale, -obj.y*p_scale))
                    is_first = False
                if xy_static and ang_static and (obj.ang != 0):
                    ang = -obj.ang*RAD_TO_DEG
                    if not is_first:
                        fid.write(' ')
                    fid.write('rotate(%.3f)' % (ang))
                    is_first = False
                if scale_static and (obj.scale != 1):
                    if not is_first:
                        fid.write(' ')
                    fid.write('scale(%.3f)' % (obj.scale*p_scale))
                elif scale_post != 1:
                    if not is_first:
                        fid.write(' ')
                    fid.write('scale(%.3f)' % (scale_post))
                fid.write('\"')
            fid.write('>\n')

            # Add the objects in this group.
            for child in obj.objs:
                add_obj(child, level + 1)

            # Add dynamic settings.
            if not xy_static:
                dynamic_translate(obj, scale_post, level + 1)
                if ang_static and (obj.ang != 0):
                    obj.ang = [obj.ang]
                    dynamic_rotate(obj, level + 1)
            if not ang_static:
                dynamic_rotate(obj, level + 1)
            if not scale_static:
                dynamic_scale(obj, scale_post, level + 1)

            # Close the group.
            fid.write('%s</g>\n' % (ind))
        elif isinstance(obj, shape):
            add_shape(obj, level)
        elif isinstance(obj, list):
            for child in obj:
                add_obj(child, level + 1)

    # Ensure 'obj' is a list.
    if isinstance(obj, (shape, frame)):
        obj = [obj]

    # Get the window width and height.
    x_span = x_lim[1] - x_lim[0]
    y_span = y_lim[1] - y_lim[0]
    win_width = 640
    win_height = round(win_width*y_span/x_span)

    # Get the overall pixel scaling factor.
    p_scale = win_width/x_span

    # Get the input x and y shifts to the middle point.
    x_shift = (x_lim[1] + x_lim[0])/2.0
    y_shift = (y_lim[1] + y_lim[0])/2.0

    # Write the file.
    with open(filename, 'w') as fid:
        fid.write('<svg viewBox=\"%d %d %d %d\"\n' %
                (-win_width/2.0, -win_height/2.0, win_width, win_height))
        fid.write('  xmlns=\"http://www.w3.org/2000/svg\">\n')
        level = 1
        if (x_shift != 0) or (y_shift != 0):
            fid.write('  <g transform=\"translate(%.3g,%.3g)\">\n' %
                    (-x_shift*p_scale, y_shift*p_scale))
            level += 1

        # Crawl through the hierarchy of frames and objects and add them to the
        # animation.
        for child in obj:
            add_obj(child, level)

        # closing
        if level > 1:
            fid.write('  </g>\n')
        fid.write('</svg>\n')
        fid.close()

# -------
# Generic
# -------

def ldiv(A, B):
    """
    Calculate the left matrix divide A\B.  It does this by using the Numpy
    LinAlg function `solve`.  This is faster than `np.linalg.inv(A).dot(B)` for
    large matrices.  For matrices of 100x100, this can be about 25% faster.  It
    is not slower for small matrices.
    """

    return np.linalg.solve(A, B)


def rdiv(A, B):
    """
    Calculate the right matrix divide A/B.  It does this by using the Numpy
    LinAlg function `solve`.  This is faster than `A.dot(np.linalg.inv(B))` for
    large matrices.  For matrices of 100x100, this can be about 25% faster.  It
    can be a little slower for small matrices.
    """

    return np.linalg.solve(B.T, A.T).T


def trim(sets, ranges, t=None, T_Sa=0):
    """
    Trim ranges of data from the `sets` of data.

    The input `ranges` should be a 2D list or ndarray of indices.  The first
    column represents the starting indices of the segments to remove and the
    second column represents the ending indices.  You can optionally provide a
    time array, `t`, in which case, the values of time after each segment will
    be shifted to occur just before the beginning of each segment.  The input
    `T_Sa` controls how much of a time step will be used in replacing the time
    gap that was created by the trimming.  If `T_Sa` is left as zero, then the
    mean time step in t will be used.
    """

    # Check input types.
    if t is not None:
        if isinstance(t, np.ndarray):
            raise Exception('trim: The first input must be an ndarray!')
    if isinstance(sets, list):
        raise Exception('trim: The second input must be a list!')
    if isinstance(ranges, np.ndarray) and isinstance(ranges, list):
        ranges = ranges.reshape(1, -1)
    if isinstance(ranges, np.ndarray) and (len(ranges.shape) == 1):
        ranges = np.array([ranges])
    elif isinstance(ranges, list):
        if isinstance(ranges[0], list):
            ranges = np.array(ranges, dtype=int)
        else:
            ranges = np.array([ranges], dtype=int)

    # Trim each range from each data set.
    if t is not None:
        # Get a default value for the time step size.
        if T_Sa == 0:
            T_Sa = np.mean(np.diff(t))

        for n_range in range(ranges.shape[0] - 1, -1, -1):
            na = ranges[n_range, 0]
            nb = ranges[n_range, 1]
            rg = np.s_[na:nb]
            t = np.concatenate((t[:na], t[nb:] - t[nb] + t[na - 1] + T_Sa))
            for n_set, a_set in enumerate(sets):
                sets[n_set] = np.delete(a_set, rg)
        return sets, t
    else:
        for n_range in range(ranges.shape[0] - 1, -1, -1):
            na = ranges[n_range, 0]
            nb = ranges[n_range, 1]
            rg = np.s_[na:nb]
            for n_set, a_set in enumerate(sets):
                sets[n_set] = np.delete(a_set, rg)
        return sets


def wc(y, x=None, modulus=np.pi):
    """
    Find the wrapped crossings of `y` and return the linearly interpolated
    corresponding values of `x`.  This function exists in part because the
    `interp` function in NumPy does not handle non-monotonically changing values
    of `xp`.  If `x` is not defined (is None), then it will be internally
    defined as an array of indices and the return value will be the real-valued
    indices of the wrapped crossings of `y`.  If no wrapped crossing can be
    found, `None` will be returned.

    Parameters
    ----------
    y : np.ndarray
        Array of y-axis values.
    x : np.ndarray, default None
        Array of x-axis values.
    modulus : float, default pi
        Modulus used to find crossings of y.

    Returns
    -------
    xc : np.ndarray
        The x-axis value of the modulus crossings of y.

    Notes
    -----
    If `y` is an angle and the default value of `modulus` is used, then this
    function will find all points where `y` crosses the pi boundary.
    """

    # Get length of y.
    N = len(y)

    # Use array of indices if x is not defined.
    if x is None:
        x = np.arange(N)

    # Wrap y.
    z = y - np.round(y/(2*modulus))*(2*modulus)

    # Find all wraps.
    nn = np.where(np.abs(np.diff(z)) > modulus)[0]

    # Interpolate to get the corresponding values of x.
    if len(nn) == 0:
        xc = None
    else:
        zc = modulus*np.sign(z[nn])
        xc = x[nn] + (x[nn + 1] - x[nn])*(zc - z[nn])/(z[nn + 1] - z[nn])

    return xc


def zc(y, x=None):
    """
    Find the zero crossings of `y` and return the linearly interpolated
    corresponding values of `x`.  This function exists in part because the
    `interp` function in NumPy does not handle non-monotonically changing values
    of `xp`.  If `x` is not defined (is None), then it will be internally
    defined as an array of indices and the return value will be the real-valued
    indices of the zero crossings of `y`.  If no zero crossing can be found,
    `None` will be returned.

    Parameters
    ----------
    y : np.ndarray
        Array of y-axis values.
    x : np.ndarray, default None
        Array of x-axis values.

    Returns
    -------
    xc : np.ndarray
        The x-axis value of the zero crossings of y.
    """

    # Get length of y.
    N = len(y)

    # Use array of indices if x is not defined.
    if x is None:
        x = np.arange(N)

    # Get indices of y where sign changes.
    isz = np.append((np.sign(y[0:(N - 1)]) != np.sign(y[1:N])), False)
    nn = np.where(isz)[0]

    # Interpolate to get the corresponding values of x.
    if len(nn) == 0:
        xc = None
    else:
        xc = x[nn] - (x[nn + 1] - x[nn])/(y[nn + 1] - y[nn])*y[nn]

    return xc


def dft(y, T_Sa=1.0):
    """
    Generate the single-sided Fourier transform of `y(t)` returning `Y(f)` in
    the order `Y`, `f`.

    Parameters
    ----------
    y : 1D np.ndarray
        Time-domain waveform array.
    T_Sa : float, default 1.0
        Sampling period in seconds.

    Returns
    -------
    Y : 1D np.ndarray
        Frequency-domain complex values of Fourier transform of y.
    f : 1D np.ndarray
        Frequency array in hertz from 0 to the Nyquist limit.

    Notes
    -----
    This function is designed such that the function ::

        y(t) = 3 + sin(2 pi t) + 7 cos(2 pi 10 t)

    will result in the following frequency components:

        =====   =========
        Freq    Magnitude
        =====   =========
        0 Hz    3
        1 Hz    1
        10 Hz   7
        =====   =========
    """

    # Get the scaled Fourier transform of y.
    Nt = len(y)
    Y = np.fft.fft(y)/Nt

    # Crop the Fourier transform to the first half of the data (below the
    # Nyquist limit) and finish the scaling.  The DC component should not be
    # doubled.
    Nt_h = np.floor(Nt/2).astype(int) + 1
    Y = Y[:Nt_h]*2
    Y[0] /= 2

    # Build the frequency array.
    df = 1/((Nt-1)*T_Sa)
    f = np.arange(Nt_h)*df

    return Y, f


def psd(y, T_Sa=1.0):
    """
    Get the power spectral density of y.

    Parameters
    ----------
    y : 1D np.ndarray
        Time-domain array of values.
    T_Sa : float, default 1.0
        Sampling period in seconds.

    Returns
    -------
    S : 1D np.ndarray
        Frequency-domain array of power spectral density values.
    f : 1D np.ndarray
        Array of frequency values from 0 to 1/T_Sa Hz.

    Notes
    -----
    See https://en.wikipedia.org/wiki/Spectral_density for a detailed definition
    of power spectral density (PSD).  The definition for discrete values comes
    down to this::

                         2
               |        |   T
        S(f) = | fft(y) |  --- ,
               |        |   N

    where T is the sampling period and N is the length of y.  The units of S(f)
    should be W/Hz.  This presumes that y^2 is instantaneous power.
    """

    Nt = len(y)
    S = np.abs(np.fft.fft(y))**2*T_Sa/Nt
    t_dur = (Nt - 1)*T_Sa
    f = np.arange(Nt)/t_dur

    return S, f


def to_step(x, y, tol=0.1, sep=0.0):
    """
    Convert a waveform defined by (x, y) to a step waveform given a tolerance.

    Parameters
    ----------
    x : 1D np.ndarray
        Input x-axis values (uniformally stepped).
    y : 1D np.ndarray
        Input y-axis values.
    tol : float, default 0.1
        Tolerance for changes in `y`.
    sep : float, default 0.0
        Separation of `x` values at steps.

    Returns
    -------
    xx : 1D np.ndarray
        Output x-axis values.
    yy : 1D np.ndarray
        Output y-axis values.

    Notes
    -----
    Consider the following example input waveform::

                             ___
        \       /\          /
         \     /  \        /
          \___/    \______/

    This would get translated to ::

        --.     .--.        .------
          |     |  |        |
          '-----'  '--------'

    Notice, the final form is just a little longer, exactly one step longer.
    """

    # Get the step size of x and the array of y changes.
    dx = x[1] - x[0]
    dy = np.abs(np.diff(y))

    # Using the changes in y, find the points that should be duplicated.
    n_dup = np.asarray(dy > tol).nonzero()[0]

    # Build an array the indices of y.
    n_keep = np.arange(len(y))

    # Combine the original array of indices with the array of indices to
    # duplicate.  Then sort these indices.
    n_y = np.sort(np.concatenate((n_keep, n_dup)))
    yy = np.append(y[n_y], y[-1])

    # Using the new arrays of indices, redefine the x and y arrays and duplicate
    # the last point.
    x_keep = x[n_keep] + sep/2
    x_dup = x[n_dup + 1] - sep/2
    xx = np.sort(np.concatenate((x_keep, x_dup)))
    xx = np.append(xx, xx[-1] + dx)

    return xx, yy


def winterp(x, xp, yp, span=6.283185307179586):
    """
    Perform wrapped linear interpolation of the input path `(xp, yp)` at the
    points `x` to produce `y`.

    Parameters
    ----------
    x : 1D np.ndarray
        Abscissa values of output array.
    xp : 1D np.ndarray
        Abscissa values of input array.
    yp : 1D np.ndarray
        Ordinate values of input array.
    span : float, default 6.283185307179586
        Range within which `y` should be wrapped.

    Returns
    -------
    y : 1D np.ndarray
        Wrap-interpolated output.

    Notes
    -----
    This function differs from NumPy's interp function in that it recognizes
    that the `y` values are wrapped.

    This function works by first using the regular NumPy interp function and
    then identifying those points where the original input path `(xp, yp)` must
    have been wrapped.  At those points, the function unwraps the input path,
    interpolates the unwrapped path, wraps the result and puts it back into the
    output array.
    """

    # Linearly interpolate the input path.
    y = np.interp(x, xp, yp)

    # Find the intervals where the input path must have wrapped.
    dyp = np.diff(yp)
    nn_jump = np.where(np.abs(dyp) >= span/2)[0]

    # At each of those intervals, unwrap the input path, interpolate, and then
    # rewrap the path.
    for nap in nn_jump:
        # Get the indices of the input path about the interval.
        nbp = nap + 1

        # Get the indices of the output path about the interval.
        na = np.where(x > xp[nap])[0][0]
        nb = np.where(x < xp[nbp])[0][-1]

        # Depending on the direction of wrapping, interpolate the path.
        if yp[nap] > yp[nbp]:
            y[na:(nb + 1)] = np.interp(x[na:(nb + 1)], [xp[nap], xp[nbp]],
                    [yp[nap], yp[nbp] + span])
        else:
            y[na:(nb + 1)] = np.interp(x[na:(nb + 1)], [xp[nap], xp[nbp]],
                    [yp[nap], yp[nbp] - span])

    # Ensure the result is properly wrapped.
    y -= np.round(y/span)*span

    return y


def find(haystack, needle):
    """
    Find the list `needle` in the list `haystack`.

    It returns the starting index of each match.  For example, if `needle` were
    the list ::

        [1, 0, 0, 0, 1, 0, 1, 1]

    and `haystack` were the list ::

        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
                     ^                                   ^

    the result would be ::

        [4, 16] .

    If `haystack` is shorter than `needle`, the empty list `[]` will be
    returned.  The normal types of the inputs and output are lists.
    """

    # Convert the inputs to lists.
    if isinstance(haystack, np.ndarray):
        haystack = list(haystack)
    if isinstance(needle, np.ndarray):
        needle = list(needle)

    # Check the lengths of the two arrays.
    N_haystack = len(haystack)
    N_needle = len(needle)
    if N_haystack < N_needle:
        return []

    # Find all matches.  A list comprehension is no faster.
    nn = []
    for n in range(N_haystack - N_needle + 1):
        if haystack[n:(n + N_needle)] == needle:
            nn.append(n)
    return nn


def sma(y, N, axis=0):
    """
    Get the simple moving average of N values of y.

    Paramters
    ---------
    y : 1D np.ndarray
        Time-domain array of values.
    N : int
        Width of the moving window.
    axis : int, default 0
        Axis along which to smooth.

    Returns
    -------
    y_bar : 1D np.ndarray
        Simple moving average of y.

    Notes
    -----
    The window in this simple moving average is centered.  See details at
    https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average.
    """

    N_pre = int(np.ceil(N/2))
    N_post = int(np.floor(N/2))

    if y.ndim == 1:
        y_pre = y[0]*np.ones(N_pre)
        y_post = y[-1]*np.ones(N_post)
        y_ext = np.concatenate((y_pre, y, y_post))
        Y = np.cumsum(y_ext)
        y_bar = (Y[N:] - Y[:-N])/N
    elif y.ndim == 2:
        if axis == 0:
            y_pre = np.ones((N_pre, 1)) @ y[0:1, :]
            y_post = np.ones((N_post, 1)) @ y[-1:, :]
            y_ext = np.concatenate((y_pre, y, y_post), axis=0)
            Y = np.cumsum(y_ext, axis=0)
            y_bar = (Y[N:, :] - Y[:-N, :])/N
        elif axis == 1:
            y_pre = y[:, 0:1] @ np.ones((1, N_pre))
            y_post = y[:, -1:] @ np.ones((1, N_post))
            y_ext = np.concatenate((y_pre, y, y_post), axis=1)
            Y = np.cumsum(y_ext, axis=1)
            y_bar = (Y[:, N:] - Y[:, :-N])/N
        else:
            raise Exception("sma: axis must be 0 or 1!")
    else:
        raise Exception("sma: y must be 1 or 2 dimensional!")

    return y_bar

# -------
# Strings
# -------

def f2str(num, width=6):
    """
    Convert a floating-point number, `num`, to a string, keeping the total width
    in characters equal to `width`.
    """

    # Ensure width is not less than 6, and check if padding should not be
    # used (i.e., width was negative).
    if width < 0:
        width = -width
        skip_padding = True
    else:
        skip_padding = False
    if width < 6:
        width = 6

    # Make num non-negative by remember the minus.
    if num < 0:
        sw = 1
        s = "-"
        num = -num
        ei = int(np.floor(np.log10(num))) # integer exponent
    elif num > 0:
        sw = 0
        s = ""
        ei = int(np.floor(np.log10(num))) # integer exponent
    else:
        sw = 0
        s = ""
        ei = 0

    # Build number string without leading spaces.
    if ei >= 4:     # 10000 to inf
        f_str = s + "%.*g" % (width - 2 - len(str(ei)) - sw, num*(10**(-ei)))
        if "." in f_str:
            f_str = f_str.rstrip("0").rstrip(".")
        f_str += "e%d" % (ei)
    elif ei >= 0:   # 1 to 10-
        f_str = s + "%.*f" % (width - 2 - ei - sw, num)
        if "." in f_str:
            f_str = f_str.rstrip("0").rstrip(".")
    elif ei >= -3:  # 0.001 to 1-
        f_str = s + "%.*f" % (width - 2 - sw, num)
        if "." in f_str:
            f_str = f_str.rstrip("0").rstrip(".")
    else:           # -inf to 0.001-
        f_str = s + "%.*g" % (width - 3 - len(str(-ei)) - sw, num*(10**(-ei)))
        if "." in f_str:
            f_str = f_str.rstrip("0").rstrip(".")
        f_str += "e%d" % (ei)

    # Add leading spaces for padding.
    if not skip_padding:
        f_str = " "*(width - len(f_str)) + f_str

    return f_str

# -----
# Files
# -----

def bin_read_s4(fstr):
    """
    Read a binary file, `fstr`, as an array of signed, 4-bit integers.  Parse
    each signed 8-bit value into two, signed, 4-bit values assuming
    little-endianness.  This means that the less significant 4 bits in an 8-bit
    value are interpreted as coming sequentially before the most significant 4
    bits in the 8-bit value.
    """

    x = np.fromfile(fstr, dtype=np.int8)
    x_even = ((np.bitwise_and(x, 0x0F) - 8) % 16) - 8
    x_odd = np.right_shift(x, 4)
    return np.array([x_even, x_odd], dtype=np.int8).flatten("F")


def bin_write_s4(x, fstr):
    """
    Write a binary file, `fstr`, as an array of signed, 4-bit integers, `x`.
    Combine pairs of two values as two, signed, 4-bit values into one 8-bit
    value, assuming little-endianness.  This means that the less significant 4
    bits in an 8-bit value are interpreted as coming sequentially before the
    most significant 4 bits in the 8-bit value.  The input x should already be
    scaled to a range of -7 to +7.  If it is not, the values will be clipped.
    """

    x[(x > 7)] = 7
    x[(x < -7)] = -7
    x = np.round(x).astype(np.int8)
    NX = int(np.ceil(len(x)/2))
    X = np.zeros(NX)
    X = np.left_shift(x[1::2], 4) + np.bitwise_and(x[0::2], 0x0F)
    X.tofile(fstr)
