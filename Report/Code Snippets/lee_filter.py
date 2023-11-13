def lee_filter(image:np.ndarray, size:int = 5) -> np.ndarray:
    """Applies lee filter to image. It is applied per channel.

    https://www.imageeprocessing.com/2014/08/lee-filter.html
    https://www.kaggle.com/code/samuelsujith/lee-filter

    Args:
        image (np.array): Unfiltered image. Example size: (2,512,512)
        size (int, optional): Kernel size (N by N). Should be odd in order to have a 'center'. Defaults to 7.
    
    Returns:
        np.ndarray: Filtered image
    """
    EPSILON = 1e-9
    filtered = np.zeros(image.shape)
    
    # Apply filter to each channel wise
    for c in range(image.shape[0]):

        avg_kernel = np.ones((size, size), np.float32) / (size**2)
        
        patch_means = cv.filter2D(image[c], -1, avg_kernel)
        patch_means_sqr = cv.filter2D(image[c]**2, -1, avg_kernel)
        patch_var = patch_means_sqr - patch_means**2

        img_var = np.mean(image[c]**2) - np.mean(image[c])**2
        patch_weights = patch_var / (patch_var + img_var + EPSILON)
        filtered[c] = patch_means + patch_weights * (image[c] - patch_means)

    return filtered