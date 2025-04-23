from collections import Counter
import numpy as np
from skimage.util import view_as_windows
from bisect import bisect_right

def first_order_entropy(image, levels=16):
    '''
        Calculates the 1st order entropy for the image per timestep
        Levels: the discrete levels/probability values

        !!! It does not account for rgb images !!!

    '''
    image = image//(256//levels)
    histogram = Counter(image.flatten())
    total_pixels = image.size
    probabilities = {i: count / total_pixels for i, count in histogram.items()}
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    return entropy




def second_order_entropy(image, levels):
    '''
        Calculates the 2nd order entropy for the image per timestep
        Levels: the discrete levels/probability values

        !!! It does not account for rgb images !!!

    '''
    def vectorized_greycomatrix(image_array, distances, angles, levels):
        max_value = levels - 1
        glcm = np.zeros((levels, levels), dtype=np.int64)

        for dist, angle in zip(distances, angles):
            # Compute the shifts based on the angle
            dx = int(round(np.cos(angle))) * dist
            dy = int(round(np.sin(angle))) * dist

            # Shift the image
            shifted_image = np.roll(image_array, shift=(dy, dx), axis=(0, 1))

            # Mask for valid (non-border) pixel comparisons
            valid_mask = np.ones_like(image_array, dtype=bool)
            if dy > 0:
                valid_mask[:dy, :] = False
            elif dy < 0:
                valid_mask[dy:, :] = False
            if dx > 0:
                valid_mask[:, :dx] = False
            elif dx < 0:
                valid_mask[:, dx:] = False

            # Extract valid pixel pairs
            i_values = image_array[valid_mask].astype(int)  # Ensure integer type
            j_values = shifted_image[valid_mask].astype(int)  # Ensure integer type

            # Accumulate counts in the GLCM
            np.add.at(glcm, (i_values, j_values), 1)
            np.add.at(glcm, (j_values, i_values), 1)  # Symmetry

        return glcm

    distances = [1]  # Pixel pair distance
    angles = [0]  # Horizontal adjacency, 0 degrees
    glcm = vectorized_greycomatrix(image, distances, angles, levels=levels)

    # Normalize the GLCM
    glcm = glcm / glcm.sum()

    # Calculate entropy as an example
    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))

    return entropy



def std_C(image, levels):
    '''
        Calculates the contrast (standard deviation of pixel values) as a metrix for information capacity calculation 

        !!! It does not account for rgb images !!!

    '''
    quantized_image = image//(256//levels)
    normalized_image = quantized_image/levels
    
    # Flatten the image
    x = normalized_image.flatten()
    
    # Calculate mean
    mean = x.mean()
    
    # Calculate histogram using numpy
    histogram, bin_edges = np.histogram(quantized_image, bins=levels, range=(0, levels))
    
    # Normalize histogram to get probabilities
    probabilities = histogram / x.size
    
    # Calculate the expected squared deviation
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / levels  # Adjust bin centers to match normalized values
    squared_deviation = ((bin_centers - mean) ** 2) * probabilities
    
    # Sum to get the weighted sum of squared deviations
    SD = np.sqrt(np.sum(squared_deviation))
    
    return SD


def window_std(image, levels, window_size=16, stride=16):
    '''
    
        !!! It does not account for rgb images !!!
    
    '''
    image = image[image.shape[0]//4:-image.shape[0]//4]/255
    # Create a view of the image with the specified window size
    windows = view_as_windows(image, (window_size, window_size, 3), step=stride)
    

    std_devs = np.std(windows, axis=(3, 4))
    mean = np.mean(windows, axis=(3, 4))
    std = np.sqrt(np.sum(std_devs**2, axis=-1))
    # print(mean.shape)
    gray = 0.5+0.5*np.tanh(5*(np.mean(mean, axis=-1)-0.5))
    distance = np.sum((mean - gray[:,:,:,np.newaxis])**2, axis = -1)
    # print(distance.shape)
    # print((std*distance).flatten())
    # print(mean[0,0], std[0,0], distance[0,0])
    # sdv

    return np.sum(std*(0.1+distance))


def C_lin(image, levels):
    '''
    
        !!! It does not account for rgb images !!!
    
    '''
    flat_image = image.flatten()
    differences = abs(flat_image[:, np.newaxis] - flat_image[np.newaxis, :])
    return np.sum(differences)/255


def C_wei(image, levels):
    '''
    
        !!! It does not account for rgb images !!!
    
    '''
    flat_image = image.flatten()
    x_flat = flat_image[:, None]  # Reshape to a column vector
    y_flat = flat_image[None, :]  # Reshape to a row vector
    differences = np.abs(x_flat - y_flat) / (x_flat + y_flat + 1e-10)  # Add epsilon to avoid division by zero
    return np.sum(differences)


def partial_contrast_with_distr(image, levels, delta = 0.0078, downsample = 1, ranges = None):
    '''
        Evaluate the partial contrast in the image (https://ieeexplore.ieee.org/document/9720755)
        delta: size of each range

        To handle RGB images, it converts them to GrayScale using the Luminance formula !
    
    '''
    if downsample > 1:
        image = image[::downsample, ::downsample]

    if image.ndim == 3:
        image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        return (partial_contrast(image[:,:,0], levels, delta, downsample, ranges) + partial_contrast(image[:,:,1], levels, delta, downsample, ranges) +
                partial_contrast(image[:,:,2], levels, delta, downsample, ranges))/3

    
    x = np.arange(0, 256).reshape(256, 1)  # Column vector
    y = np.arange(0, 256).reshape(1, 256)  # Row vector

    # Apply the formula abs(x - y) / (x + y)
    c_matrix = np.abs(x - y) / 255

    hist, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    def neighbours(image, k):
        x_values_h = image[:, :-k].ravel()  # All rows, all columns except the last
        y_values_h = image[:, k:].ravel()    # All rows, all columns except the first

        # Extract vertical pairs (current pixel and below neighbor)
        x_values_v = image[:-k, :].ravel()   # All rows except the last, all columns
        y_values_v = image[k:, :].ravel()     # All rows except the first, all columns

        # Combine horizontal and vertical pairs
        x_values = np.concatenate((x_values_h, x_values_v))
        y_values = np.concatenate((y_values_h, y_values_v))

        # Calculate the 2D histogram of neighboring pixels
        hist2d, xedges, yedges = np.histogram2d(x_values, y_values, bins=256, range=[[0, 256], [0, 256]])
        return hist2d
    
    hist2d = neighbours(image, 1)
    hist2d += neighbours(image, 4)
    hist2d += neighbours(image, 16)
    hist2d += neighbours(image, 64)
    hist2d /=4

    # Normalize the histogram to get probabilities (optional)
    probability_hist2d = hist2d / hist2d.sum()
    # probabilities_over_hist = probability_hist2d * hist
    ranges = [(0, delta), (delta, 10 * delta), (10 * delta, 25 * delta), (25 * delta, 35 * delta), (35 * delta, 1)] if ranges is None else ranges
    def get_interval_indices(values, ranges):
        range_starts = np.array([r[0] for r in ranges])
        range_ends = np.array([r[1] for r in ranges])
        indices = np.searchsorted(range_starts, values, side='right') - 1
        valid_mask = (values >= range_starts[indices]) & (values < range_ends[indices])
        indices[~valid_mask] = -1
        return indices

    distr = np.zeros(len(ranges))
    indices = get_interval_indices(c_matrix, ranges)
    np.add.at(distr, indices, probability_hist2d)

    optimal_distr = [0.32-0.16, 0.42+0.16, 0.12+0.08, 0.08-0.03, 0.06-0.01]

    return distr    # do this to only get the distr of images
    # return np.sum(np.square(optimal_distr-distr))









def partial_contrast(image, levels, delta = 0.0078, downsample = 1, ranges = None):
    '''
        Evaluate the partial contrast in the image (https://ieeexplore.ieee.org/document/9720755)
        delta: size of each range

        To handle RGB images, it converts them to GrayScale using the Luminance formula !
    
    '''

    image = image[image.shape[0]//4:-image.shape[0]//4]/255
    image_gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    image /= np.sqrt(3)    
    def neighbours(image, k, ranges):
        x_values_h = image[:, :-k, :]  # All rows, all columns except the last
        y_values_h = image[:, k:, :]    # All rows, all columns except the first
        x_values_h_gray = image_gray[:, :-k]  # All rows, all columns except the last
        y_values_h_gray = image_gray[:, k:]    # All rows, all columns except the first

        # Extract vertical pairs (current pixel and below neighbor)
        x_values_v = image[:-k, :, :]   # All rows except the last, all columns
        y_values_v = image[k:, :, :]     # All rows except the first, all columns
        x_values_v_gray = image_gray[:-k, :]   # All rows except the last, all columns
        y_values_v_gray = image_gray[k:, :]     # All rows except the first, all columns


        # x_values_v_gray = 0.5+0.5*np.tanh(5*(x_values_v_gray -0.5))
        # y_values_v_gray = 0.5+0.5*np.tanh(5*(y_values_v_gray -0.5))
        # x_values_h_gray = 0.5+0.5*np.tanh(5*(x_values_h_gray -0.5))
        # y_values_h_gray = 0.5+0.5*np.tanh(5*(y_values_h_gray -0.5))
        
        
        
        
        x_values_h = image[:, :-k, :]  # All rows, all columns except the last
        # print(((1 - 9/2*np.sum((x_values_h-x_values_h_gray[:, :, np.newaxis]) ** 2, axis=2)) * (1 - 9/2*np.sum((y_values_h-y_values_h_gray[:, :, np.newaxis]) ** 2, axis=2)) > 0.999))
        # print(x_values_h_gray)
        # print(x_values_h)
        # sd
        c_matrix_h = np.sqrt(np.sum((x_values_h - y_values_h) ** 2, axis=2)) *   (2*(((1 - np.sum((x_values_h-x_values_h_gray[:, :, np.newaxis]) ** 2, axis=2)) * (1 - np.sum((y_values_h-y_values_h_gray[:, :, np.newaxis]) ** 2, axis=2)) > 0.99) -0.5))
        c_matrix_v = np.sqrt(np.sum((x_values_v - y_values_v) ** 2, axis=2)) *   (2*(((1 - np.sum((x_values_v-x_values_v_gray[:, :, np.newaxis]) ** 2, axis=2)) * (1 - np.sum((y_values_v-y_values_v_gray[:, :, np.newaxis]) ** 2, axis=2)) > 0.99) -0.5))
        distr, _ = np.histogram(c_matrix_h, bins=ranges)
        distr += np.histogram(c_matrix_v, bins=ranges)[0]

        return distr/image.shape[0]/image.shape[1]/2
    
    ranges = [0, delta, 5 * delta, 15 * delta, 35 * delta, 1] if ranges is None else ranges
    
    
    # distr = neighbours(image, 1, ranges)
    # distr += neighbours(image, 4, ranges)
    distr = neighbours(image, 16, ranges)
    distr += neighbours(image, 64, ranges)
    distr /=2
    # print(distr)
    # ds
    
    optimal_distr = [0.32-0.16, 0.42+0.16, 0.12+0.08, 0.08-0.03, 0.06-0.01]

    return distr    # do this to only get the distr of images