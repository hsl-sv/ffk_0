def _ffkloop_np(data_target, epdxf, epdyf,
                nslow, nchan, chans,
                FK):

    dn_div = (nchan - 1) / nchan

    # Calculate FK for each x,y slowness in a loop
    for sx in range(nslow):

        # Get phase delays for this x slowness
        sxx = sx * nchan + chans

        # Delay data for this x slowness
        epdxfsxx = np.multiply(data_target, epdxf[:,sxx])

        for sy in range(nslow):

            # Get phase delays for this y slowness
            syy = sy * nchan + chans

            # Incorporate delays for this y slowness
            pdata = np.multiply(epdxfsxx, epdyf[:,syy])

            # Sum delayed data for each frequency
            # then square and sum magnitude for all frequnecies
            pd_sum = np.sum(pdata, axis=1)
            FK[sx, sy] = np.sum(abs2(pd_sum) ** 2)

    return FK

def ffk(DATA, SAMPRATE, X, Y, SLOW, BAND):
    
    nslow = np.size(np.array(SLOW, ndmin=2), 1)
    nchan = np.size(np.array(DATA, ndmin=2), 0)

    # Calculate the FT of 'data'
    DATA_FT = np.fft.fft(DATA)
    DATA_FT = np.real(DATA_FT) + (-1j * np.imag(DATA_FT))
    m = np.size(DATA_FT, 1)

    # Compute the frequency bands
    dfreq = SAMPRATE / m
    lfreq = np.int32(np.ceil((BAND[0]) / dfreq))
    hfreq = np.int32((np.floor(BAND[1]) / dfreq))

    # Extract the arrays
    FK = np.zeros((nslow, nslow))
    fk = np.real(FK)
    slow = np.real(SLOW)

    chans = np.arange(0, len(DATA))
    f = np.transpose(np.arange(lfreq, hfreq + 1)[np.newaxis])
    data_target = DATA_FT[:, f[:, 0]].T

    slow = np.transpose(np.array(slow)[np.newaxis])
    X = np.array([X])
    Y = np.array([Y])
    pdx = np.dot(-slow, X)
    pdy = np.dot(-slow, Y)

    epdxfs1 = (-1j * 2 * np.pi * dfreq * pdx).flatten()[np.newaxis]
    epdxfs2 = np.dot(f + 1, epdxfs1)
    epdxf = np.exp(epdxfs2)
    epdyfs1 = (-1j * 2 * np.pi * dfreq * pdy).flatten()[np.newaxis]
    epdyfs2 = np.dot(f + 1, epdyfs1)
    epdyf = np.exp(epdyfs2)

    FK = _ffkloop_np(data_target, epdxf, epdyf, nslow, nchan, chans, FK)

    return FK
