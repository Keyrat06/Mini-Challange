from scipy import ndimage
import numpy as np
def scrambleImages(images):
    for i in range(len(images)):
        
        # 15% chance that nothing is done
        if np.random.rand() < 0.15:
            pass
         
        if np.random.rand() < 0.5: # Flip horizontally
            images[i] = np.fliplr(images[i])
            
        if np.random.rand() < 0.5: # Rotate
            angle = (np.random.rand()-0.5)*2*15
            images[i] = ndimage.rotate(images[i], angle, reshape=False)
        if np.random.rand() < 0.5: # Blur image
            images[i] = ndimage.gaussian_filter(images[i], 3)
    return images
