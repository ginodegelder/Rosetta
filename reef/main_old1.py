import numpy as np

from . import processes as prc
from . import tools_old1 as tools


def reef(dbg, t, e, tmax, u0, G0, zmax, zmin, zow, slopi, v0, zb, dx):
    """
    Reef function

    Parameters
    ----------
    dbg : bool
        Used for debugging

    t : numpy array
        Time in the sea-level curve (My)

    e : numpy array
        Value of the sea-level curve

    tmax : float
        Max time evolution in My

    u0 : float
        ?? Uplift ??

    G0 : float
        Reef growth rate

    zmax : float
        Max depth for reef growth

    zmin :
        Optimal depth for reef growth

    zow :
        ??

    slopi : float
        Initial slope

    v0: float
        Erosion rate

    zb : float
        ??

    dx : float
        ??

    Returns
    -------
    x, y, dz, xmin, xmax, ymin, ymax, x_ini, y_ini
    """
    # if dbg == True:
    # dt=t_in[0]-t_in[1]                         # dt in ky
    dt = 1
    # else:
    #    dt=tools.defdt(zmin, G0, dx, np.max(e_in[1:]-e_in[:-1])-u0)
    #     t=np.arange(t_in[0], t_in[-1]-dt, -dt)
    #     e=np.flip(np.interp(np.flip(t), np.flip(t_in), np.flip(e_in)))

    # Defining the number of iterations
    if tmax == -1:
        tmax = t[0]
    #     tpt=t[0]-1
    #     imax=int(tmax/dt)
    imax = int(tmax)

    G = G0 * dt
    u = u0 * dt
    upmax = u * tmax
    dz = upmax
    # u=np.zeros(imax+1)
    # u[:]=u0*dt
    # upmax=np.sum(u[1:])

    # Initial topography
    x, y, jmax, xmin, xmax, ymin, ymax = tools.geominidx(slopi, dx, upmax,
                                                         np.min(e))

    x_ini, y_ini = np.copy(x), np.copy(y)

    ###########################################################################

    # Time loop
    for i in range(1, imax + 1):
        # if dbg == True:
        # print("\nTime loop", "i", i, "t", t[i], "ky", "e", e[i])
        # print('\n')

        if i == imax:
            debug = True
        else:
            debug = False

        # Vertical motion
        y = y + u
        # dz=upmax-np.sum(u[1:i+1])
        dz = dz - u

        ######################################################################

        # Reef growth

        # if debug:
        #     test = open("oldprof", 'w')
        #     np.savetxt(test, (np.c_[x, y]), fmt='%1.2f')
        #     test.close()

        # x, y, prems, der = prc.const(
        # debug, t[i], zow, G, zmin, zmax, e[i], x, y)

        ######################################################################

        # Wave erosion
        y, prems, der = prc.erosion(u, v0, zb, e[i - 1], e[i], dx, y)

        # if debug:
        #     test = open("test0", 'w')
        #     np.savetxt(test, (np.c_[x, y]), fmt='%1.8f')
        #     test.close()

        x, y = tools.resampdx(debug, prems, der, dx, x, y)

        # if i == imax:
        #     test = open('test1', 'w')
        #     np.savetxt(test, (np.c_[x, y]), fmt='%1.2f')
        #     test.close()

        #######################################################################

        if prems <= 100:
            jmax, prems, der, x, y = tools.raboutprems(prems, der, x, y)
        if der >= jmax - 500:
            jmax, prems, der, x, y = tools.raboutder(prems, der, x, y)

        xmax = max(xmax, x[der] - 1)
        xmin = min(xmin, x[prems])
        ymax = max(ymax, y[der] + dz - 1)
        ymin = min(ymin, y[prems] + dz)

    return x, y, dz, xmin, xmax, ymin, ymax, x_ini, y_ini

## Input parameters
# zmax=20.            # Max depth for reef growth
# zmin=1              # Optimal depth for reef growth
# zow=2
# tmax=-1
# zb=3
# V0=20
# G0=0

# nb=0

# curves=['SL-sinus-asym2.dat', 'waelbroeck2002.dat', "SL-static.dat"]

# for slopi in range (2, 16, 4):            # Initial slope
##     print('slope', slopi)
#    for V0 in range (10, 100, 10):                # Max reef growth rate
##         print('G0', G0)
#        for u0 in range(0, 10, 1):
#            u0=u0/10
##             print('u0', u0)
#            
#            for slcurve in curves:
#                #slcurve=(slcurve[0:-1])
##                 print(slcurve)

#                for dx in [1]:    #, 5, 10, 100]:
#                    #for tmax in [10, 20]:

#                    dit=main(False, slcurve, tmax, u0, G0, zmax, zmin, zow, slopi, V0, zb, dx)
#                        #main(True, slcurve, tmax, u0, G0, zmax, zmin, zow, slopi, 1)
#                    if dit!=9999:
#                        if nb==0:
#                            dmin=dit
#                            dmax=dit
#                            dmoy=dit
#                        nb=nb+1
#                        dmin=min(dmin, dit)
#                        dmax=max(dmax, dit)
#                        dmoy=dmoy+dit
#                    
#                        print('dmin', dmin)
#                        print('dmax', dmax)
#                        print('dmoy', dmoy, dmoy/nb)


# # Input parameters
# zmax=20.            # Max depth for reef growth
# zmin=5              # Optimal depth for reef growth
# zow=2

# slcurve="SL-static.dat"
# #slcurve="waelbroeck-500y.dat"
# #slcurve="waelbroeck2002.dat"

# #slcurve="SL-sinus-asym2.dat"
# #slcurve="SL-sinus-asym-500y.dat"
# # slcurve="SL-sinus-asym-100y.dat"
# dx=100
# u0=0
# G0=10
# slopi=5

# tmax=10

# main(True, slcurve, tmax, u0, G0, zmax, zmin, zow, slopi, dx)
