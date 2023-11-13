def fast_frost_filter(image, d=2.0, k=5):
    """Applies Frost filter to the image, per channel

    Args:
        image (np.ndarray): Image in shape C
        d (float, optional): Dampening Factor. Defaults to 2.0.
        k (int, optional): Kernel size (k by k). Defaults to 5.

    Returns:
        np.ndarray: Filtered image
    """
    assert k%2==1

    mean_filter = np.ones( (k,k) ) / (k**2)
    filtered_img = np.zeros(image.shape)
    for c in range(image.shape[0]):
        
        scene = image[c]
        mean = cv.filter2D(scene, -1, kernel=mean_filter)
        var = cv.filter2D(scene**2, -1, kernel=mean_filter) - mean**2
        
        # Create the distance from center pixel window
        distances = np.zeros( (k,k) )
        ce = np.floor(k/2)  # (c,c) is the center pixel array index
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distances[i,j] = np.sqrt((i-ce)**2 + (j-ce)**2)
        
        distances = np.reshape(distances, (25,1,1))
    
        b = d * (var / (mean * mean + 1e-9) )
        b = np.broadcast_to(b, (k**2, scene.shape[0], scene.shape[1]))
        x = np.zeros( shape=(k**2, scene.shape[0], scene.shape[1]) )

        W = np.exp(-b * distances)
        
        for n in range(k**2):
            x[n] = cv.filter2D(scene, -1, subwindow_kernel(n, k))
        
        filtered_img[c] = np.sum(x*W, axis=0) / np.sum(W, axis=0)
    
    return filtered_img