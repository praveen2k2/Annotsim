import numpy as np

def generate_value_noise(shape, scale=16):
    # Generate random values for the grid points
    grid = np.random.rand(*shape)

    # Interpolate the values using bilinear interpolation
    x, y = np.meshgrid(np.arange(shape[1]) / scale, np.arange(shape[0]) / scale)
    x_int, y_int = x.astype(int), y.astype(int)
    x_frac, y_frac = x - x_int, y - y_int

    x1 = x_int % shape[1]
    x2 = (x1 + 1) % shape[1]
    y1 = y_int % shape[0]
    y2 = (y1 + 1) % shape[0]

    # Bilinear interpolation
    value_noise = (1 - x_frac) * (1 - y_frac) * grid[y1, x1] + \
                  x_frac * (1 - y_frac) * grid[y1, x2] + \
                  (1 - x_frac) * y_frac * grid[y2, x1] + \
                  x_frac * y_frac * grid[y2, x2]

    return value_noise

def generate_multi_octave_3d_value_noise(shape, octaves=2, persistence=0.9, frequency=1):
    # Initialize the noise grid
    noise = np.zeros(shape)
    amplitude = 1
    # Generate value noise for each octave and each channel
    for channel in range(shape[0]):
        value_noise = np.zeros(shape[1:])
        for _ in range(octaves):
            # Generate value noise for the current octave
            octave_noise = generate_value_noise(shape[1:], scale=frequency)
            value_noise += amplitude * octave_noise
#             frequency /= 2
            amplitude *= persistence

        # Normalize each channel to the range [0, 1]
        noise[channel] = value_noise #(value_noise - value_noise.min()) / (value_noise.max() - value_noise.min())#value_noise#

    return noise
