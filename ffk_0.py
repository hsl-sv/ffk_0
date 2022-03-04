import numpy as np
from numba import jit

@jit(nopython=True)
def _ffkloop(data_FT_r, data_FT_i, fk,
             dfreq, lfreq, hfreq,
             slow, x, y, cos_x, cos_y, sin_x, sin_y,
             nslow, nchan):
    for f in range(lfreq, hfreq):
        index = 0

        # Pre-compute the sin and cos terms
        for i in range(0, nslow):
            for k in range(0, nchan):
                tmp = -2 * np.pi * (f + 1.0) * dfreq * x[k] * slow[i]

                cos_x[index] = np.cos(tmp)
                sin_x[index] = np.sin(tmp)

                tmp = -2 * np.pi * (f + 1.0) * dfreq * y[k] * slow[i]

                cos_y[index] = np.cos(tmp)
                sin_y[index] = np.sin(tmp)

                index += 1

        # Loop over each x,y slowness pair
        for i in range(0, nslow):

            for j in range(0, nslow):
                beam_value_r = 0
                beam_value_i = 0

                i_index = i * nchan
                j_index = j * nchan

                # Loop over each element in the array
                for k in range(0, nchan):
                    cos_tmp = cos_x[i_index] * cos_y[j_index] - sin_x[i_index] * sin_y[j_index]
                    sin_tmp = sin_x[i_index] * cos_y[j_index] + cos_x[i_index] * sin_y[j_index]

                    index = f

                    beam_value_r += data_FT_r[k][index] * cos_tmp - data_FT_i[k][index] * sin_tmp
                    beam_value_i += data_FT_i[k][index] * cos_tmp + data_FT_r[k][index] * sin_tmp

                    i_index += 1
                    j_index += 1

                fk[i][j] += beam_value_r * beam_value_r + beam_value_i * beam_value_i

    return fk


def ffk(DATA, SAMPRATE, X, Y, SLOW, BAND):
    """Fast F-K Analysis

        Parameters
        ----------
        DATA : 2D array
            Input data array, multiple sites
        SAMPRATE : int
            Sampling rate of DATA
        X : list
            dx
        Y : list
            dy
        SLOW : list
            Slowness list
        BAND : list
            List contains bandpass filter bounds

        Returns:
        ----------
        fk_result : 2D array
            Slowness X * Slowness Y size F-K analysis result, Azimuth starts at East, clockwise

    """

    nslow = np.size(np.array(SLOW, ndmin=2), 1)
    nchan = np.size(np.array(DATA, ndmin=2), 0)

    # Calculate the FT of 'data'
    DATA_FT = np.fft.fft(DATA)
    m = np.size(DATA_FT, 1)

    # Compute the frequency bands
    dfreq = SAMPRATE / m
    lfreq = np.int32(np.ceil((BAND[0]) / dfreq))
    hfreq = np.int32((np.floor(BAND[1]) / dfreq) + 1)

    # Extract the arrays
    FK = np.zeros((nslow, nslow))
    fk = np.real(FK)
    slow = np.real(SLOW)

    data_FT_r = np.real(DATA_FT)
    data_FT_i = np.imag(DATA_FT)

    x = np.real(X)
    y = np.real(Y)

    # Preallocate the sin and cos terms
    cos_x = np.zeros(nslow * nchan)
    sin_x = np.zeros(nslow * nchan)
    cos_y = np.zeros(nslow * nchan)
    sin_y = np.zeros(nslow * nchan)

    # Loop over each frequency point
    fk_result = _ffkloop(data_FT_r, data_FT_i,
                         fk, dfreq, lfreq, hfreq, slow,
                         x, y,
                         cos_x, cos_y, sin_x, sin_y,
                         nslow, nchan)

    return fk_result

if __name__ == '__main__':

    # Read your file
    wfdisc, site, data = [], [], []

    # Your configuration
    band_low = 1
    band_high = 19
    band = [1.0, 19.0]
    slow = np.linspace(-400, 400, 80, endpoint=True)
    samprate = 100
    x = []
    y = []

    FK = ffk(data, samprate, x, y, slow, band)
