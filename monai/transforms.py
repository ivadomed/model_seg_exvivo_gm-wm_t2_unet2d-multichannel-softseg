import numpy as np
import torch
import scipy.ndimage as ndimage
import numpy.linalg as linalg
import nibabel as nib

from monai.data import MetaTensor
from monai.transforms import Spacing, Affine, ScaleIntensity

def rotMatToAxisAngles(rotmat):
    """Given a ``(3, 3)`` rotation matrix, decomposes the rotations into
    an angle in radians about each axis.
    """

    yrot = np.sqrt(rotmat[0, 0] ** 2 + rotmat[1, 0] ** 2)

    if np.isclose(yrot, 0):
        xrot = np.arctan2(-rotmat[1, 2], rotmat[1, 1])
        yrot = np.arctan2(-rotmat[2, 0], yrot)
        zrot = 0
    else:
        xrot = np.arctan2( rotmat[2, 1], rotmat[2, 2])
        yrot = np.arctan2(-rotmat[2, 0], yrot)
        zrot = np.arctan2( rotmat[1, 0], rotmat[0, 0])

    return [xrot, yrot, zrot]

def decompose(xform, angles=True, shears=False):
    """Decomposes the given transformation matrix into separate offsets,
    scales, and rotations, according to the algorithm described in:

    Spencer W. Thomas, Decomposing a matrix into simple transformations, pp
    320-323 in *Graphics Gems II*, James Arvo (editor), Academic Press, 1991,
    ISBN: 0120644819.

    It is assumed that the given transform has no perspective components.

    :arg xform:  A ``(3, 3)`` or ``(4, 4)`` affine transformation matrix.

    :arg angles: If ``True`` (the default), the rotations are returned
                 as axis-angles, in radians. Otherwise, the rotation matrix
                 is returned.

    :arg shears: Defaults to ``False``. If ``True``, shears are returned.

    :returns: The following:

               - A sequence of three scales
               - A sequence of three translations (all ``0`` if ``xform``
                 was a ``(3, 3)`` matrix)
               - A sequence of three rotations, in radians. Or, if
                 ``angles is False``, a rotation matrix.
               - If ``shears is True``, a sequence of three shears.
    """

    # The inline comments in the code below are taken verbatim from
    # the referenced article, [except for notes in square brackets].

    # The next step is to extract the translations. This is trivial;
    # we find t_x = M_{4,1}, t_y = M_{4,2}, and t_z = M_{4,3}. At this
    # point we are left with a 3*3 matrix M' = M_{1..3,1..3}.
    xform = np.array(xform).T

    if xform.shape == (4, 4):
        translations = xform[ 3, :3]
        xform        = xform[:3, :3]
    else:
        translations = np.array([0, 0, 0])

    M1 = xform[0]
    M2 = xform[1]
    M3 = xform[2]

    # The process of finding the scaling factors and shear parameters
    # is interleaved. First, find s_x = |M'_1|.
    sx = np.sqrt(np.dot(M1, M1))
    M1 = M1 / sx

    # Then, compute an initial value for the xy shear factor,
    # s_xy = M'_1 * M'_2. (this is too large by the y scaling factor).
    sxy = np.dot(M1, M2)

    # The second row of the matrix is made orthogonal to the first by
    # setting M'_2 = M'_2 - s_xy * M'_1.
    M2 = M2 - sxy * M1

    # Then the y scaling factor, s_y, is the length of the modified
    # second row.
    sy = np.sqrt(np.dot(M2, M2))

    # The second row is normalized, and s_xy is divided by s_y to
    # get its final value.
    M2  = M2  / sy
    sxy = sxy / sx

    # The xz and yz shear factors are computed as in the preceding,
    sxz = np.dot(M1, M3)
    syz = np.dot(M2, M3)

    # the third row is made orthogonal to the first two rows,
    M3 = M3 - sxz * M1 - syz * M2

    # the z scaling factor is computed,
    sz = np.sqrt(np.dot(M3, M3))

    # the third row is normalized, and the xz and yz shear factors are
    # rescaled.
    M3  = M3  / sz
    sxz = sxz / sx
    syz = syz / sy

    # The resulting matrix now is a pure rotation matrix, except that it
    # might still include a scale factor of -1. If the determinant of the
    # matrix is -1, negate the matrix and all three scaling factors. Call
    # the resulting matrix R.
    #
    # [We do things different here - if the rotation matrix has negative
    #  determinant, the flip is encoded in the x scaling factor.]
    R = np.array([M1, M2, M3])
    if linalg.det(R) < 0:
        R[0] = -R[0]
        sx   = -sx

    # Finally, we need to decompose the rotation matrix into a sequence
    # of rotations about the x, y, and z axes. [This is done in the
    # rotMatToAxisAngles function]
    if angles: rotations = rotMatToAxisAngles(R.T)
    else:      rotations = R.T

    retval = [np.array([sx, sy, sz]), translations, rotations]

    if shears:
        retval.append(np.array((sxy, sxz, syz)))

    return tuple(retval)

class ToMetaTensor:
    """
    A class to convert a nibabel image to a MONAI MetaTensor with optional axis swapping.
    """

    def __init__(self, swap=None):
        """
        Initializes the transform.

        Args:
            swap (tuple of ints, optional): Axis swap order.
        """
        self.swap = swap

    def __call__(self, nib_path):
        """
        Apply the transformation to a nibabel image.

        Args:
            nib_path (str): The path to the nibabel image.

        Returns:
            MetaTensor: The transformed image as a MetaTensor.
        """
        nib_image = nib.load(nib_path)
        image_data = nib_image.get_fdata()
        image_affine = nib_image.affine.copy()

        if self.swap is not None:
            image_data = np.transpose(image_data, self.swap)
            image_affine[:3, 3] = [image_affine[:3, 3][self.swap[0]], 
                                   image_affine[:3, 3][self.swap[1]], 
                                   image_affine[:3, 3][self.swap[2]]]

        if image_data.ndim == 3:
            image_data = image_data[np.newaxis, ...]

        tensor = torch.tensor(image_data, dtype=torch.float32)

        ### metadata

        spacing = np.array(nib_image.header.get_zooms())    
        if self.swap is not None:
            spacing = spacing[list(self.swap)]

        filename = nib_image.get_filename()
        if "gmseg" in filename or "wmseg" in filename:
            type = "mask"
        else:
            type = "image"

        meta_dict = {"affine": image_affine, 
                     "shape": image_data.shape,
                     "spacing": tuple(spacing),
                     "type": type}

        return MetaTensor(tensor, meta=meta_dict)

class SmoothResampling:
    def __init__(self, resampling, smoothing=True):
        """
        Initialize the SmoothResampling object.
        
        Args:
            resampling (np.ndarray): The target voxel spacing.
            smoothing (bool): Whether to apply smoothing.
        """
        self.resampling = resampling
        self.smoothing = smoothing

    def applySmoothing(self, meta_tensor, matrix, newShape):
        """Applies smoothing to the data."""
        ratio = decompose(matrix[:3, :3])[0] / np.array(meta_tensor.meta["spacing"])

        if len(newShape) > 3:
            ratio = np.concatenate((
                ratio,
                [float(o) / float(s)
                 for o, s in zip(meta_tensor.meta["shape"][3:], newShape[3:])]))

        sigma = np.array(ratio)
        sigma[ratio <  1.1] = 0
        sigma[ratio >= 1.1] *= 0.425

        smoothed_data = ndimage.gaussian_filter(meta_tensor.numpy()[0, :, :, :], sigma)    
        smoothed_data = smoothed_data[np.newaxis, ...]

        smoothed_tensor = torch.tensor(smoothed_data, dtype=torch.float32)

        return MetaTensor(smoothed_tensor, meta=meta_tensor.meta)

    def __call__(self, meta_tensor):
        """Performs resampling and optionally smoothing on the input MetaTensor."""
        if type(self.resampling) == np.ndarray and self.smoothing:
            p = meta_tensor.meta["spacing"]
            shape = meta_tensor.meta["shape"][1:]

            ratio = np.array(self.resampling) / np.array(p)
            shape_r = tuple([int(np.round(shape[i] * float(p[i]) / float(self.resampling[i]))) for i in range(min(meta_tensor.ndim, 3))])

            affine = meta_tensor.meta["affine"].numpy()[:4, :4]
            affine[3, :] = np.array([0, 0, 0, 1])

            R = np.eye(4)
            for i in range(3):
                R[i, i] = meta_tensor.meta["shape"][i+1] / float(shape_r[i])

            affine_r = np.dot(affine, R)
            meta_smooth = self.applySmoothing(meta_tensor, affine_r, shape_r)
        else:
            meta_smooth = meta_tensor

        if type(self.resampling) == np.ndarray:
            resample_transform = Spacing(pixdim=self.resampling, recompute_affine=False)
            meta_resample = resample_transform(meta_smooth, output_spatial_shape=shape_r)
            return meta_resample
        else:
            return meta_smooth

class RandAffineRel:
    def __init__(self, prob=0.5, affine_translation=(0.1, 0.1), affine_degrees=45, padding_mode="zeros"):
        """
        Initialize the RandAffineRel object with specified parameters.

        Args:
            prob (float): Probability of applying the transform.
            affine_translation (tuple of float): Maximum translation as a fraction of image dimensions.
            affine_degrees (float): Maximum rotation in degrees.
            padding_mode (str): Padding mode ('zeros', 'border', or 'reflection').
        """
        self.prob = prob
        self.affine_translation = affine_translation
        self.affine_degrees = affine_degrees
        self.padding_mode = padding_mode

        #random float between 0 and 1
        rd = np.random.rand()

        if rd < self.prob:
            self.affine_translation = (np.random.rand() * self.affine_translation[0], 
                                       np.random.rand() * self.affine_translation[1])
            self.affine_degrees = np.random.rand() * self.affine_degrees

        else :
            self.affine_translation = (0, 0)
            self.affine_degrees = 0

    def __call__(self, slice):
        """
        Apply randomized affine transformations to the input slice.

        Args:
            slice: The slice to transform.

        Returns:
            The transformed slice.
        """
        # Calculate translate_range based on the shape of the slice and affine_translation
        
        # add channel dimebsion if needed
        if len(slice.shape) == 2:
            slice = slice[np.newaxis, ...]

        # Calculate translation range based on the shape of the slice and affine_translation
        translation = (self.affine_translation[0] * slice.shape[1], 
                              self.affine_translation[1] * slice.shape[2])

        # Convert degrees to radians for rotate_range
        rotate_angle = self.affine_degrees * np.pi / 180

        # Create RandAffine transform with specified ranges and probability
        affine_transform = Affine(rotate_params=rotate_angle, 
                                      translate_params=translation, padding_mode=self.padding_mode, image_only=True)        
        # Apply the transform
        slice_randaffine = affine_transform(slice)

        return slice_randaffine

def remove_empty_slices(images, masks):

    valid_indices = [i for i, (image, mask) in enumerate(zip(images, masks)) if not (image.sum() == 0 or mask.sum() == 0)]
    filtered_images = [images[i] for i in valid_indices]
    filtered_masks = [masks[i] for i in valid_indices]
    return filtered_images, filtered_masks
