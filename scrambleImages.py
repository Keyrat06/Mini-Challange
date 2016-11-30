from scipy import ndimage
import numpy as np
#import matplotlib.pyplot as plt
def scrambleImages(images):
    for i in range(len(images)):
        
        # 15% chance that nothing is done
        if np.random.rand() < 0.2:
            pass
        if np.random.rand() < 0.4: # Crop and resize
            topleft_x = np.random.randint(0,33)
            topleft_y = np.random.randint(0,33)
            height = np.random.randint(64,97)
            width = np.random.randint(64,97)
            temp = images[i][topleft_x:topleft_x+width, topleft_y:topleft_y+height,:]
            temp = ndimage.zoom(temp.astype('int32'), [128.0/width,128.0/height,1])
            temp = temp.astype('float32')
            images[i] = temp
        if np.random.rand() < 0.5: # Flip horizontally
            images[i] = np.fliplr(images[i])
        if np.random.rand() < 0.3: # Rotate
            angle = (np.random.rand()-0.5)*2*30
            images[i] = ndimage.rotate(images[i], angle, reshape=False)
        if np.random.rand() < 0.2: # Blur image
            images[i] = ndimage.gaussian_filter(images[i], 3)
        if np.random.rand() < 0.2: # Adjust contrast randomly
            temp = images[i]
            mean_val = np.mean(temp)
            images[i] = (temp-mean_val)*(np.random.rand()*20/2+0.5)+mean_val

            
    return images
