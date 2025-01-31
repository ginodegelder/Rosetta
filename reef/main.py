from datetime import datetime as dtime

import numpy as np

from reef import tools as tools
from .context import jitmode


@jitmode
def reef(t_in, e_in, tmax, u0, slopi, V0, zb, dx, dt0, xmini, xmaxi, ymini,
         ymaxi):
    """
    Reef function. Updated 14/04/2021.

    Parameters
    ----------
    t_in :  numpy array
        Time in the sea-level curve (My).

    e_in : numpy array
        Value of the sea-level curve.

    tmax : float
        Max time evolution in My.

    u0 : float
        ?? Uplift ??

    slopi : float
        Initial slope.

    V0 : float
        Erosion rate.

    zb : float
        ??

    dx : float
        Spatial discretization.

    dt0 : float
        Time discretization.

    xmini
    xmaxi
    ymini
    ymaxi

    Returns
    -------

    """

    beta1 = 0.1  # Coefficient for erosion efficiency, sea bed abrasion

    # Defining timestep and number of iterations
    imax, dt = tools.defdt_eros(dt0, V0, beta1, tmax, t_in)

    # Resampling SL with dt
    t = np.arange(t_in[0], t_in[-1] - dt, -dt)
    e = np.flip(np.interp(np.flip(t), np.flip(t_in), np.flip(e_in)))

    # Scaling parameters to the time step
    V = V0 * dt
    u, upmax = tools.u_story(imax, dt, u0)

    # Initial topography
    x, y, jmax, tmp, tmp, tmp, tmp = tools.geominidx(slopi, dx, upmax,
                                                     np.min(e))
    x_ini, y_ini = np.copy(x), np.copy(y)
    cr_mem = np.zeros(x.size)  # Cumulative cliff retreat,
    # records remaining Vrest to cumulate over several iterations

    #######################################################################

    # Time loop
    for i in range(0, imax):

        # Vertical motion
        y = y + u[i]
        dz = upmax - np.sum(u[:i + 1])

        # SL erosion
        y, prems, der = tools.erosion(u, V, zb, beta1, e[i], dx, y, cr_mem)
        tools.rabout(prems, der, x, y)

        xmini, xmaxi, ymini, ymaxi = tools.adjust_limits(
            prems, der, xmini, xmaxi, ymini, ymaxi, dz, x, y)

    #######################################################################

    return x, y, dz, xmini, xmaxi, ymini, ymaxi, x_ini, y_ini, dt
