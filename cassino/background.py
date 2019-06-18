import matplotlib.pyplot as plt
import numpy as np

import lightkurve as lk
import fbpca
from tqdm import tqdm
from astropy.stats import sigma_clipped_stats
from photutils import Background2D, MedianBackground


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

    def __init__(self, tpf, nterms=50):
        self.tpf = tpf
        self._validate()
        self._remove_frame_background()
        self.nterms = nterms

    def __repr__(self):
        return 'cassino.BackgroundModel'

    def _validate(self):
        ''' Checks if the inputs are valid'''
        if not isinstance(self.tpf, lk.targetpixelfile.TargetPixelFile):
            raise CassinoException('Please pass a lightkurve.targetpixelfile.TargetPixelFile object.')

        self.flux = self.tpf.flux
        self.error = self.tpf.flux_err
        self.time = self.tpf.time

        if len(self.flux.shape) != 3:
            raise CassinoException('`flux` must be 3D.')

        if self.flux.shape != self.error.shape:
            raise CassinoException('`flux` must be the same size as `error`.')


    def _remove_frame_background(self, max_flux=5e2):
        mask = np.median(self.flux, axis=0) < max_flux
        s = np.asarray([sigma_clipped_stats(f[mask]) for f in self.flux])
        self.flux -= np.atleast_3d(s[:, 1]).transpose([1, 0, 2])


    def _build_components(self, nterms):
        pix = self.flux[:, np.ones(self.flux.shape[1:], bool)]
        components, _, _ = fbpca.pca(pix, nterms)
        self.components = components

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
                ax.plot(self.components[:, comp])
                ax.set_ylabel('Component')
            ax.set_xlabel('Time')
        return fig

    def _compute_stellar_model(self):
        '''Stellar model. Uses PCA to remove common trends.'''

        stellar_model = np.zeros(self.flux.shape)
        weights = np.zeros((self.nterms, self.flux.shape[1], self.flux.shape[2]))
        for i in tqdm(range(self.flux.shape[1]), desc='building stellar model'):
            for j in range(self.flux.shape[2]):
                f, fe = np.copy(self.flux[:, i, j]), self.error[:, i, j]
                f -= sigma_clipped_stats(f)[1]
                A = np.dot(self.components.T, self.components/fe[:, None]**2)
                B = np.dot(self.components.T, f[:, None]/fe[:, None]**2)
                weights[:, i, j] = np.linalg.solve(A, B).reshape(self.nterms)
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


    def _compute_mask(self, correction=None):
        ''' Builds a mask for pixels that are poorly corrected '''
        if correction is None:
            correction = np.zeros(self.flux.shape)

        s = np.zeros((3, self.flux.shape[1], self.flux.shape[2]))
        for idx in tqdm(range(self.flux.shape[1]), desc='masking'):
            for jdx in range(self.flux.shape[2]):
                    s[:, idx, jdx] = np.asarray([sigma_clipped_stats(self.flux[:, idx, jdx] - correction[:, idx, jdx])])
        s1 = sigma_clipped_stats(s[0])
        im_mask = np.abs(s[0] - s1[1]) >  (5 * s1[2])
        s1 = sigma_clipped_stats(s[2])
        im_mask |= np.abs(s[2] - s1[1]) >  (10 * s1[2])
        return im_mask

    def compute(self):
        ''' Build the background model '''

        # Build PCA components
        self._build_components(nterms=self.nterms)

        self.stellar_model, self.stellar_weights = self._compute_stellar_model()
        self.strap_model =  self._compute_strap_model(correction=self.stellar_model)

        self.scatter_model = self._compute_scatter_model(correction=self.stellar_model + self.strap_model)
        self.model = self.stellar_model + self.strap_model + self.scatter_model

        self.corrected_flux = self.flux - self.model
        mask = self._compute_mask(correction=self.stellar_model)
        self.corrected_flux[:, mask] = 0
        return
