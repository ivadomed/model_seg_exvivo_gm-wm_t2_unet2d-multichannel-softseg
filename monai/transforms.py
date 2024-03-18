import numpy as np
from monai.transforms import Affine

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
