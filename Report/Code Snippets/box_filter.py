def box_filter(image:np.ndarray, size:int=5) -> np.ndarray:
    """Applies box filter to an image

    Args:
        image (np.ndarray): Input image
        size (int, optional): Kernel size. Defaults to 5.

    Returns:
        np.ndarray: _description_
    """
    avg_kernel = np.ones( shape=(size,size), dtype=np.float32) / (size**2)
    return cv.filter2D(image, -1, avg_kernel)