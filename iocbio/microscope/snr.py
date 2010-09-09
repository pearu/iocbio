import numpy

__all__ = ['estimate_snr']

def estimate_snr(image,
                 noise_type = 'poisson',
                 use_peak = False):
    """
    Estimate signal-to-noise ratio of the image stack.

    Signal-to-noise ratio is defined as follows::

      SNR = E(image) / (sqrt(E(image - E(image))^2)

    where E(image) is the averaged image. In theory, calculating
    averaged image requires multiple acquisitions of the same image.
    E(image) can be estimated by smoothing the image with a gaussian.

    Parameters
    ----------
    image : `iocbio.io.image_stack.ImageStack`

    noise_type : {'poisson'}

    Notes
    -----

    In the case of Poisson noise the signal-to-noise ratio
    is defined as peak signal-to-noise ratio:

      SNR = sqrt(E(max image)).

    """
    if noise_type=='poisson':
        if use_peak:
            mx = image.max()
        else:
            data = image
            values = []
            for indices in zip (*numpy.where(data==data.max())):
                for i0 in range (indices[0]-1,indices[0]+2):
                    if i0>=data.shape[0]:
                        i0 -= data.shape[0]
                    for i1 in range (indices[1]-1,indices[1]+2):
                        if i1>=data.shape[1]:
                            i1 -= data.shape[1]
                        for i2 in range (indices[2]-1,indices[2]+2):
                            if i2>=data.shape[2]:
                                i2 -= data.shape[2]
                            values.append (data[i0,i1,i2])
            mx = numpy.mean(values)
        snr = numpy.sqrt(mx)
    else:
        raise NotImplementedError(`noise_type`)
    return snr
    
