import numpy as np

import cv2 as cv

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.draw import line

from scipy.signal import convolve2d


class Hog:

    def __init__(self, k, block_size):
        if k < 1:
            raise ValueError("El factor de escala k debe ser mayor o igual a 1.")
        
        self.k = k
        self.block_size = block_size
        
    def convert_rgb_to_gray(self, image):
        """Converts the image from RGB to grayscale.
        
        Args:
            image (np.ndarray): The input RGB image to be converted.
        
        Returns:
            np.ndarray: The grayscale image.
        """
        return rgb2gray(image)
    
    def resize_image_ratio_2_1(self, image):
        """Resizes the image to a 2:1 ratio based on the given scale factor.
        The new size will be (64 * k, 128 * k), where k is the scale factor.

        Args:
            image (np.ndarray): The input image to be resized.
            
        Returns:
            np.ndarray: The resized image.
        """
        return resize(image, (128 * self.k, 64 * self.k))
        
    def __pad_image__(self, image, pad_width=1, mode="symmetric", reflect_type="even"):
        """Pads the image with the specified width and mode.

        Args:
            image (np.ndarray): The input image to be padded.
            pad_width (int): The width of the padding. Default is 1.
            mode (str): The mode for padding. Default is "symmetric".
            reflect_type (str): The type of reflection for padding. Default is "even".

        Returns:
            np.ndarray: The padded image.
        """
        return np.pad(image, pad_width=pad_width, mode=mode, reflect_type=reflect_type)
    
    def __gaussian_filter__(self, image, kernel_size=3, sigma=1):
        """Applies a Gaussian filter to the image.

        Args:
            image (np.ndarray): The input image to be filtered.
            kernel_size (int): The size of the Gaussian kernel. Default is 3.
            sigma (float): The standard deviation of the Gaussian distribution. Default is 1.
            
        Returns:
            np.ndarray: The filtered image.
        """
        
        kernel = cv.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        kernel /= np.sum(kernel)
        
        filtered_image = convolve2d(image, kernel, mode="same")
        
        return filtered_image
        
    def calculate_image_gradient(self, image):
        """Calculates the image gradient using Prewitt filters.
        
        Args:
            image (np.ndarray): The input image to calculate the gradient.

        Returns:
            tuple: A tuple containing the gradient magnitude and gradient angle.
                - gradient_magnitude (np.ndarray): The gradient magnitude of the image.
                - gradient_angle (np.ndarray): The gradient angle of the image.
        """
        # Prewitt filters for gradient calculation
        hx = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ])
        
        hy = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])
        
        # Apply Gaussian filter to the image to reduce noise
        gaussian_filter = self.__gaussian_filter__(image, kernel_size=31, sigma=0.03)
        
        # Pad the image to handle borders
        padded_image = self.__pad_image__(gaussian_filter)
        
        gradient_x = convolve2d(padded_image, hx, mode="valid")
        gradient_y = convolve2d(padded_image, hy, mode="valid")

        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_angle = (np.arctan2(-gradient_y, gradient_x) * (180 / np.pi)) % 180
        
        return gradient_magnitude, gradient_angle
    
    def build_block_stack(self, image):
        """Builds a stack of blocks from the image array.

        Args:
            image (np.ndarray): The input image array.

        Returns:
            np.ndarray: The stack of blocks.
        """
        block_stack = []
        
        stride = self.block_size // 2
        M, N = image.shape
        
        for i in range(0, M - self.block_size + 1, stride):
            for j in range(0, N - self.block_size + 1, stride):
                block = image[i:i + self.block_size, j:j + self.block_size]
                block_stack.append(block)
        
        return np.array(block_stack)
    
    def calculate_histogram(self, magnitude, angle, num_bins=9):
        """Calculates the histogram of gradients for each block.
        
        The histogram is normalized to unit length.
        
        Args:
            magnitude (np.ndarray): The gradient magnitude of the image.
            angle (np.ndarray): The gradient angle of the image.
            num_bins (int): The number of bins for the histogram. Default is 9.
            
        Returns:
            np.ndarray: The normalized histogram of gradients for each block.
        """
        histograms = []
        MN = magnitude.shape[0]
        bin_width = 180 / num_bins

        for i in range(MN):
            mu_block = magnitude[i].ravel()
            ot_block = angle[i].ravel()
            
            block_hist = np.zeros(num_bins)

            for k in range(len(mu_block)):
                m = mu_block[k]
                theta = ot_block[k]

                bin_idx = int(theta // bin_width)
                bin_idx_next = (bin_idx + 1) % num_bins

                alpha = (theta - (bin_idx * bin_width)) / bin_width

                block_hist[bin_idx] += m * (1 - alpha)
                block_hist[bin_idx_next] += m * alpha
            
            norm = np.linalg.norm(block_hist)
            if norm > 0:
                block_hist /= norm
            
            histograms.append(block_hist)

        return np.array(histograms)
    
    def build_directional_histogram(self, image, histograms, num_bins=9, block_threshold=0.01, angle_min=10.0, angle_max=170.0):
        """Builds a directional histogram from the image.
        
        The histogram is normalized to unit length.
        
        Args:
            image (np.ndarray): The input image.
            histograms (np.ndarray): The histogram of gradients for each block.
            num_bins (int): The number of bins for the histogram. Default is 9.
            block_threshold (float): The threshold for block normalization. Default is 0.01.
            angle_min (float): The minimum angle for the histogram. Default is 10.0.
            angle_max (float): The maximum angle for the histogram. Default is 170.0.
            
        Returns: 
            np.ndarray: The directional histogram of the image.
        """
        height, width = image.shape
        stride = self.block_size // 2
        
        M = (height - self.block_size) // stride + 1
        N = (width - self.block_size) // stride + 1
        histograms = histograms.reshape(M, N, num_bins)

        direction_image = np.zeros((height, width), dtype=np.float32)
        bin_width = 180.0 / num_bins
        bin_centers = np.linspace(bin_width / 2, 180 - bin_width / 2, num_bins)

        for m in range(M):
            for n in range(N):
                histogram = histograms[m, n]
                if histogram.sum() < block_threshold:
                    continue

                center_y = m * stride + self.block_size // 2
                center_x = n * stride + self.block_size // 2

                for b in range(num_bins):
                    magnitude = histogram[b]

                    angle_deg = bin_centers[b]
                    if not (angle_min <= angle_deg <= angle_max):
                        continue

                    angle_rad = np.deg2rad(angle_deg)
                    length = (self.block_size // 2 - 1) * magnitude

                    dx = int(round(length * np.cos(angle_rad)))
                    dy = int(round(length * np.sin(angle_rad)))

                    x1 = center_x - dx
                    y1 = center_y - dy
                    x2 = center_x + dx
                    y2 = center_y + dy

                    rr, cc = line(y1, x1, y2, x2)
                    rr = np.clip(rr, 0, height - 1)
                    cc = np.clip(cc, 0, width - 1)

                    direction_image[rr, cc] += 1.0

        return np.clip(direction_image, 0, 1)
