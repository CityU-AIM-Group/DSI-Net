import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class RandomElasticTransform(object):
    """Randomly rotate image"""
    # https://gist.github.com/nasimrahaman/8ed04be1088e228c21d51291f47dd1e6
    def __init__(self, alpha =2000, sigma=50):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
            
        shape = img.shape[:2]
        random_state = np.random        
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                             self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                             self.sigma, mode="constant", cval=0) * self.alpha    
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        
        image = map_coordinates(img, indices, order=1, mode='nearest').reshape(shape)
            
        return image
    


    