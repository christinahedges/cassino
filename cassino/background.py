import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import lightkurve as lk
import fbpca
from tqdm import tqdm
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Gaussian2DKernel
from photutils import Background2D, MedianBackground
from scipy.ndimage import gaussian_filter

from itertools import combinations_with_replacement as multichoose

import warnings

class CassinoException(Exception):
    """Raised if there is a problem with cassino."""
    pass

class BackgroundModel(object):
    ''' Models background

    Parameters
    ----------
    data : np.ndarray or lightkurve.TargetPixelFile
        3D data array of flux values

    TODO add support for not a tpf.
    '''

    def __init__(self, tpf, nterms=50, raw=False):
        self.tpf = tpf
        self._validate(raw=raw)
#        self._remove_frame_background()
        self.nterms = nterms

    def __repr__(self):
        return 'cassino.BackgroundModel'

    def _validate(self, raw=False):
        ''' Checks if the inputs are valid'''
        if not isinstance(self.tpf, lk.targetpixelfile.TargetPixelFile):
            raise CassinoException('Please pass a lightkurve.targetpixelfile.TargetPixelFile object.')

        l = lambda x: np.polyval(np.polyfit(self.tpf.hdu[1].data['RAW_CNTS'][0].ravel(), self.tpf.flux[0].ravel(),  1), x)


        if raw:
            self.flux = l(self.tpf.hdu[1].data['RAW_CNTS'])
            self.error = self.flux**0.5
            self.time = self.tpf.hdu[1].data['TIME']

            mask = (np.nansum(self.flux, axis=(1, 2)) != 0)
            mask &= self.tpf.quality_mask
            self.flux = self.flux[mask]
            self.error = self.error[mask]
            self.time = self.time[mask]
        else:
            self.flux = self.tpf.flux
            self.error = self.tpf.flux_err
            self.time = self.tpf.time

        self.flux -= sigma_clipped_stats(self.flux)[1]


        if len(self.flux.shape) != 3:
            raise CassinoException('`flux` must be 3D.')

        if self.flux.shape != self.error.shape:
            raise CassinoException('`flux` must be the same size as `error`.')


    def _remove_frame_background(self, max_flux=5e2):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = sigma_clipped_stats(self.flux.ravel())
        mask = np.median(self.flux, axis=0) < s[1]+s[2]*3
        if not mask.any():
            mask = np.ones(self.flux.shape[1:])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = np.asarray([sigma_clipped_stats(f[mask]) for f in self.flux])
        self.flux -= np.atleast_3d(s[:, 1]).transpose([1, 0, 2])


    def _build_components(self, nterms):
        pix = self.flux[:, np.ones(self.flux.shape[1:], bool)]
        components, _, _ = fbpca.pca(pix, nterms)


        c, r = self.tpf.estimate_centroids(aperture_mask='all')
        A = lambda X, Y, T: np.array([X**4, X**3, X**2, X,
                           Y**4, Y**3, Y**2, Y,
                           X**4*Y**3, X**4*Y**2, X**4*Y, X**3*Y**2, X**3*Y, X**2*Y, X*Y,
                           Y**4*X**3, Y**4*X**2, Y**4*X, Y**3*X**2, Y**3*X, Y**2*X,
                           np.ones(X.shape),
                           T, T**2, T**3]).T
        components = np.hstack([components, A(c, r, self.time)])
        #
        # matrix = np.product(list(multichoose(components.T, 2)), axis=1).T
        # matrix, _, _ = fbpca.pca(matrix, nterms)
        # components = np.hstack([components, matrix])
        #
        # matrix = np.product(list(multichoose(components.T, 3)), axis=1).T
        # matrix, _, _ = fbpca.pca(matrix, nterms)
        # components = np.hstack([components, matrix])


        self.components = components
        self.ncomponents = self.components.shape[1]


    def plotAsteroidMask(self, **kwargs):
        '''Plot the asteroid mask, to check that nasty tracks have been removed.'''

        z, x, y = (~self._asteroid_mask).nonzero()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, c='k', marker='.', s=0.5, label='Masked Pixels')
        ax.legend()
        ax.set_xlabel('Time [frame number]')
        ax.set_ylabel('Column [pix]')
        ax.set_zlabel('Row [pix]')
        return ax

    def plotComponents(self, ncomps=5):
        ''' Plots the components in an easy to understand format

        Parameters:
        -----------
        ncomps: int
            Number of components to plot up to
        '''
        if not hasattr(self, 'components'):
            raise CassinoException("Please run the compute method first.")

        with plt.style.context(lk.MPLSTYLE):
            # The components in order of the most powerful across all pixels
            order = np.argsort(np.nansum(np.abs(self.stellar_weights), axis=(1,2)))[::-1]
            fig = plt.figure(figsize=(4*2.5, ncomps*2.5))
            for idx, comp in enumerate(order[0:ncomps]):
                v = np.max(np.abs([np.percentile(self.stellar_weights[comp], 1), np.percentile(self.stellar_weights[comp], 99)]))
                ax = plt.subplot2grid((ncomps, 4), (idx, 3))
                if idx == 0:
                    ax.set_title("Weights")
                ax.imshow(self.stellar_weights[comp], vmin=-v, vmax=v)
                ax = plt.subplot2grid((ncomps, 4), (idx, 0), colspan=3)
                ax.plot(self.components[:, comp], label=comp)
                ax.legend()
                ax.set_ylabel('Component')
            ax.set_xlabel('Time')
        return fig

    def _compute_stellar_model(self, correction=None, mask=None):
        '''Stellar model. Uses PCA to remove common trends.'''
        if correction is None:
            correction = np.zeros(self.flux.shape)

        if mask is None:
            mask = np.ones(self.flux.shape, bool)

        stellar_model = np.zeros(self.flux.shape)
        weights = np.zeros((self.ncomponents, self.flux.shape[1], self.flux.shape[2]))
        for i in tqdm(range(self.flux.shape[1]), desc='building stellar model'):
            for j in range(self.flux.shape[2]):
                f, fe = np.copy(self.flux[:, i, j]), self.error[:, i, j]
                k = (f != 0) & np.isfinite(f)
                k &= mask[:, i, j]
                if k.sum() < 5:
                    raise CassinoException('Too many masked values.')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    f -= sigma_clipped_stats(f[k])[1]

                A = np.dot(self.components[k].T, self.components[k]/fe[k][:, None]**2)
                B = np.dot(self.components[k].T, f[k][:, None]/fe[k][:, None]**2)
                weights[:, i, j] = np.linalg.solve(A, B).reshape(self.ncomponents)
                stellar_model[:, i, j] = np.dot(self.components, weights[:, i, j].reshape(-1, 1))[:, 0]

        corr = np.atleast_3d(np.nanmedian(self.flux - stellar_model, axis=0)).transpose(2, 0, 1)
        stellar_model += corr
        return stellar_model, weights


    def _compute_strap_model(self, correction=None, mask=None, poly_order=2):
        ''' Strap model. Fits a polynomial in the row direction only.'''
        if correction is None:
            correction = np.zeros(self.flux.shape)

        if mask is None:
            mask = np.ones(self.flux.shape, bool)

        model = np.zeros(self.flux.shape)
        for tdx, c, e, m in tqdm(zip(range(len(self.time)), self.flux - correction, self.error, mask), desc='building strap model', total=len(self.flux)):
            l = np.zeros(c.shape)
            goodness_of_fit = np.zeros(len(c.T))
            for idx, c1, e1, m1 in zip(range(len(c.T)), c.T, e.T, m.T):
                if m1.sum() <= 5:
                    continue
                l[:, idx] = np.polyval(np.polyfit(np.arange(len(c1))[m1], c1[m1], poly_order, w=e1[m1]), np.arange(len(c1)))
                fl = (np.sum(c1**2/e1**2))
                pf = (np.sum((c1 - l[:, idx])**2/e1**2))
                goodness_of_fit[idx] = fl/pf
            l[:, goodness_of_fit < 2] = 0
            model[tdx, :, :] = l
        return model

    def _compute_scatter_model(self, correction=None):
        ''' Scatter model. Built using background 2d from photutils'''
        if correction is None:
            correction = np.zeros(self.flux.shape)
        return np.asarray([Background2D(d, [5, 5]).background
                                        for d in tqdm(self.flux - correction, total=len(self.time), desc='building scatter model')])


    # def _compute_mask(self, correction=None):
    #     ''' Builds a mask for pixels that are poorly corrected '''
    #     if correction is None:
    #         correction = np.zeros(self.flux.shape)
    #
    #     s = np.zeros((3, self.flux.shape[1], self.flux.shape[2]))
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #     for idx in tqdm(range(self.flux.shape[1]), desc='masking'):
    #         for jdx in range(self.flux.shape[2]):
    #                 s[:, idx, jdx] = np.asarray([sigma_clipped_stats(self.flux[:, idx, jdx] - correction[:, idx, jdx])])
    #     s1 = sigma_clipped_stats(s[0])
    #     im_mask = np.abs(s[0] - s1[1]) >  (5 * s1[2])
    #     s1 = sigma_clipped_stats(s[2])
    #     im_mask |= np.abs(s[2] - s1[1]) >  (10 * s1[2])
    #     return im_mask

    def _compute_asteroid_outlier_mask(self, correction):
        '''Finds a mask in a corrected cube, where there are extremely bright asteroids.'''

        b = np.copy(self.flux - correction)
        # Whiten by pixel time series std dev
        s = np.atleast_3d(np.nanstd(b, axis=(0))).transpose([2, 0, 1])

        # Sum in one dimension, remove median and find 3 sigma outliers
        ysum = np.nanmax(gaussian_filter(np.nan_to_num(b/s), (1.5, 1.5, 0)), axis=1)
        ymed = np.atleast_2d(np.nanmedian(ysum, axis=1)).T * np.ones(ysum.shape)
        yans = ysum - ymed > 3

        # Sum in other dimension, remove median and find 3 sigma outliers
        xsum = np.nanmax(gaussian_filter(np.nan_to_num(b/s), (1.5, 0, 1.5)), axis=2)
        xmed = np.atleast_2d(np.nanmedian(xsum, axis=1)).T * np.ones(xsum.shape)
        xans = xsum - xmed > 3

        # Convolve with 45 degree gaussians, to make sure if we miss a time stamp we have a shot of getting it.
        c1 = convolve(yans.astype(float), Gaussian2DKernel(1, 0.2, theta=-45))
        c2 = convolve(yans.astype(float), Gaussian2DKernel(1, 0.2, theta=45))
        yans = np.any([(c1 > 0.01), (c2 > 0.01)], axis=0)
        c1 = convolve(xans.astype(float), Gaussian2DKernel(1, 0.2, theta=-45))
        c2 = convolve(xans.astype(float), Gaussian2DKernel(1, 0.2, theta=45))
        xans = np.any([(c1 > 0.01), (c2 > 0.01)], axis=0)

        # Find where these two dimensions cross
        a = (np.atleast_3d(xans) | np.atleast_3d(yans).transpose([0, 2, 1]))

        # Weak Gaussian blur helps find real sources
        c = gaussian_filter(np.nan_to_num(b/s), (0, 0.5, 0.5))

        # Where there are sources in the image
        threshold = (c > 5*np.atleast_3d(np.nanstd(c, axis=(1,2))).transpose([1, 0, 2]))

        # Take any sources that are in the crosshairs and have high SNR.
        aper = threshold & a
        return ~aper

    def compute(self, strap=True, scatter=True):
        ''' Build the background model '''

        # Build PCA components
        self._build_components(nterms=self.nterms)

        mask = np.ones(self.flux.shape, bool)
        for iters in [0, 1]:
            self.stellar_model, self.stellar_weights = self._compute_stellar_model(mask=mask)
            self.model = self.stellar_model
            if strap:
                self.strap_model =  self._compute_strap_model(correction=self.stellar_model, mask=mask)
                self.model += self.strap_model
            if scatter:
                self.scatter_model = self._compute_scatter_model(correction=self.stellar_model + self.strap_model)
                self.model += self.scatter_model
            # Temporarily mask out bad stars...
            if iters == 0:
                self._asteroid_mask = self._compute_asteroid_outlier_mask(correction=self.model)


                s = np.zeros((3, self.flux.shape[1], self.flux.shape[2]))
                for idx in tqdm(range(self.flux.shape[1]), desc='building outlier mask'):
                    for jdx in range(self.flux.shape[2]):
                        f = self.flux[:, idx, jdx] - self.model[:, idx, jdx]
                        f[~self._asteroid_mask[:, idx, jdx]] = np.nan
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            s[:, idx, jdx] = sigma_clipped_stats(f)
                self._outlier_mask = np.abs((self.flux - self.model) - np.atleast_3d(s[1]).transpose([2, 0, 1])) < 10 * np.atleast_3d(s[2]).transpose([2, 0, 1])
                mask = self._asteroid_mask & self._outlier_mask

        self.corrected_flux = self.flux - self.model
#        mask = self._compute_mask(correction=self.stellar_model)
#        self.corrected_flux[:, mask] = 0
        return
