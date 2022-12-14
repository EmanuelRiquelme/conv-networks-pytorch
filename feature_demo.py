import numpy as np
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt
import skimage.measure

def padding(img = np.random.rand(4,4),padding_size = (1,1)):
    print(f'original:\n{img}\npadded:\n{np.pad(img,1)}')

class conv_components:
    def __init__(self,img_name = 'mr_incredible.jpg',max_pool_factor = 3,loc=0,scale=1):
        self.kernel = np.random.laplace(loc=0.0, scale=1.0, size=9).reshape(3,3)
        self.img = Image.open(f'img/{img_name}')
        self.input_img = np.asarray(self.img).transpose(2,0,1)/255
        self.factor = max_pool_factor
        self.feature_map = self.__feature_map__()

    def __feature_map__(self):
        conv = np.array([signal.convolve2d(self.input_img[chanel_input], self.kernel, boundary='symm', mode='same') for chanel_input in range(3)])
        return  conv.transpose(1,2,0).astype(np.uint8)*255

    def __max_pool__(self,img):
        return skimage.measure.block_reduce(img, block_size=(1,self.factor,self.factor), func=np.max).transpose(1,2,0)

    def visualize(self):
        fig, (ax_img,ax_maxpool,ax_feature_map,ax_feature_max_pool) = plt.subplots(4, 1, figsize=(6, 15))
        #original img
        ax_img.imshow(np.asarray(self.img), cmap='gray')
        ax_img.set_title('Original')
        ax_img.set_axis_off()
        #maxpool
        ax_maxpool.imshow(self.__max_pool__(self.input_img), cmap='gray')
        ax_maxpool.set_title('maxpool')
        ax_maxpool.set_axis_off()
        #feature_map
        ax_feature_map.imshow(self.feature_map, cmap='gray')
        ax_feature_map.set_title('feature map')
        ax_feature_map.set_axis_off()
        #feature map maxpool
        ax_feature_max_pool.imshow(self.__max_pool__(self.feature_map.transpose(2,0,1)), cmap='gray')
        ax_feature_max_pool.set_title('feature map and max pool')
        ax_feature_max_pool.set_axis_off()
 
        plt.show()

if __name__ == "__main__":
    padding()
    components = conv_components()
    print(f'kernel:\n{components.kernel}')
    components.visualize()
    print(components.stride())
