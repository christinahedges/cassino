import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize
from tqdm import tqdm
from . import PACKAGEDIR
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def movie(dat, out='out.mp4', title='', scale='linear', facecolor='red', **kwargs):
    '''Create an mp4 movie of a 3D array

    Parameters
    ----------
        dat : np.ndarray
            3D data to plot
        out : str
            Output filename
        title : str
            Title to write
        scale : str
            Linear or log scale
        facecolor : str
            Color of the background, NaNs will be this color.
        **kwargs : **dict
            Keywords to pass to matplotlib
    '''
    if scale == 'log':
        data = np.log10(np.copy(dat))
    else:
        data = dat
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.set_facecolor(facecolor)
    if 'vmax' not in kwargs:
        kwargs['vmax'] = np.nanpercentile(data, 75)
    if 'vmin' not in kwargs:
        kwargs['vmin'] = np.nanpercentile(data, 5)
    im1 = ax.imshow(data[0], origin='bottom', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=15)
    cbar1 = fig.colorbar(im1, ax=ax)
    cbar1.ax.tick_params(labelsize=10)
    if scale == 'log':
        cbar1.set_label('log$_{10}$ Flux [e$^-$s$^{-1}$]', fontsize=10)
    else:
        cbar1.set_label('Flux [e$^-$s$^{-1}$]', fontsize=10)

    def animate(i):
        im1.set_array(data[i])
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=30)
    anim.save(out, dpi=150)
