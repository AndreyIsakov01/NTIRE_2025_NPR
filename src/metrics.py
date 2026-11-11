import numpy as np
from scipy.ndimage import gaussian_filter


def psnr_metric(image1: np.ndarray,
                image2: np.ndarray,
                max_pixel_value: int | float = 255) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    PSNR is a measure of the quality of a reconstructed image compared to the original.
    Higher PSNR values indicate better image quality.

    Parameters
    ----------
    image1 : np.ndarray
        The original image as a numpy array with dtype np.uint8.
    image2 : np.ndarray
        The noisy or degraded image as a numpy array with dtype np.uint8.
    max_pixel_value : int or float, optional
        The maximum possible pixel value in the image. Default is 255.

    Returns
    -------
    float
        The PSNR value in decibels (dB). Returns np.inf if the images are identical
        (i.e., when MSE equals 0).

    Raises
    ------
    TypeError
        If either image1 or image2 is not of dtype np.uint8.
    ValueError
        If image1 and image2 do not have the same shape, or if
        max_pixel_value <= 0.

    Notes
    -----
    The function converts the original image to float64 to avoid overflow during
    computation. The PSNR formula is: PSNR = 20 * log10(MAX_I / sqrt(MSE)),
    where MAX_I is the maximum pixel value and MSE is the Mean Squared Error.
    """
    if image1.shape != image2.shape:
        raise ValueError('Both images must have the same dimensions')

    if not (isinstance(max_pixel_value, (int, float)) and max_pixel_value > 0):
        raise ValueError('max_pixel_value must be a positive number')

    # Avoiding np.uint8 overflow
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return np.inf

    PSNR_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return PSNR_value


def ssim_metric(image1: np.ndarray,
                image2: np.ndarray,
                dynamic_range: float = 255,
                truncate: float = 3.5,
                sigma: float = 1.5,
                k1: float = 0.01,
                k2: float = 0.03) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Parameters
    ----------
    image1 : np.ndarray
        Reference image (dtype=np.uint8).
    image2 : np.ndarray
        Compared image with same shape and dtype as image1.
    dynamic_range : float, optional
        The dynamic range of the pixel values (default 255).
    truncate : float, optional
        Truncate parameter for Gaussian filter (default 3.5).
    sigma : float, optional
        Standard deviation for Gaussian filter (default 1.5).
    k1 : float, optional
        Constant for SSIM calculation (default 0.01 [1]).
    k2 : float, optional
        Constant for SSIM calculation (default 0.03 [1]).

    Returns
    -------
    float
        Mean SSIM value over image channels with values in [-1, 1].

    Raises
    ------
    TypeError
        If either input is not np.uint8.
    ValueError
        If inputs do not have the same shape or dynamic_range is not positive.

    Notes
    -----
    This implementation follows the original SSIM formulation using a Gaussian
    window (implemented via scipy.ndimage.gaussian_filter). Stability constants
    K1=0.01 and K2=0.03 are used to form C1 and C2.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
        (2004). Image quality assessment: From error visibility to
        structural similarity. IEEE Transactions on Image Processing,
        13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        :DOI:`10.1109/TIP.2003.819861`
    """
    if image1.shape != image2.shape:
        raise ValueError('Both images must have the same dimensions')

    if dynamic_range <= 0:
        raise ValueError('dynamic_range must be a positive number')

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    win_size = 2 * int(truncate * sigma + 0.5) + 1

    ssim_values = []
    for channel_idx in range(image1.shape[2]):
        image1_channel = image1[..., channel_idx]
        image2_channel = image2[..., channel_idx]

        mu_x = gaussian_filter(image1_channel, sigma=sigma, truncate=truncate)
        mu_y = gaussian_filter(image2_channel, sigma=sigma, truncate=truncate)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        sigma_x2 = gaussian_filter(image1_channel ** 2, sigma=sigma, truncate=truncate) - mu_x2
        sigma_y2 = gaussian_filter(image2_channel ** 2, sigma=sigma, truncate=truncate) - mu_y2
        sigma_xy = gaussian_filter(image1_channel * image2_channel, sigma=sigma, truncate=truncate) - mu_xy
        c1 = (k1 * dynamic_range) ** 2
        c2 = (k2 * dynamic_range) ** 2
        ssim_map = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))

        ignore_offset = (win_size - 1) // 2
        cropped_ssim_map = ssim_map[ignore_offset:-ignore_offset, ignore_offset:-ignore_offset]
        ssim_value = np.mean(cropped_ssim_map)
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)