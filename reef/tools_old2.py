import math
from io import StringIO
import numpy as np

from .context import jitmode


# ********************************************************************************************
@jitmode
def defdt_eros(dt, V0, beta, tmax, t):
    # Calculates dt and defines the number of iterations
    # If dt!=0, the value provided as input sets the time step, otherwise
    # dt is calculated here
    if dt == 0:
        dt = 0.1 / (V0 * math.exp(-4) * beta)

    # Defining the number of iterations
    if tmax == -1:  # Simulation through the whole SL curve
        tmax = t[0]
    imax = int(tmax / dt)

    return imax, dt


# *************************************************************************
@jitmode
def erosion(u, V, zb, beta1, e, dx, y, cr_mem):
    # Based on Anderson et al., 1999
    Vtmp = 0
    beta2 = 1  # Coefficient for erosion efficiency, cliff retreat
    hnotch = 1  # Height of notch for volume eroded during cliff retreat

    # Variables initialisation
    deb = np.argmax(e - y <= zb)  # First node for wave erosion
    riv = np.argmax(e <= y)  # Last node for sea bed erosion
    Vrest = V

    # Vertical sea-bed erosion
    for j in range(deb, riv):
        dh = e - y[j]  # Water height
        if dh <= zb and Vrest > 0:
            dV = min(Vrest * math.exp(-dh / (zb / 4)) * beta1, Vrest / dx)
            y[j] = y[j] - dV
            Vrest = Vrest - dV * dx

    # Cliff retreat
    cr_mem[riv] = cr_mem[riv] + Vrest
    if cr_mem[riv] >= dx * hnotch:  # Enough eroded volume at this node for
        # cliff retreat
        clr = int(cr_mem[riv] / (hnotch * beta2 * dx))  # Number of nodes
        # exposed to cliff retreat
        cr_mem[riv] = cr_mem[riv] + np.sum(cr_mem[riv + 1:riv + clr + 1])
        while cr_mem[riv] - clr * hnotch * dx > hnotch * dx:
            # print('Ca depasse !!!')
            clr = clr + 1
            Vtmp = Vtmp + cr_mem[clr]
        y[riv:riv + clr] = e - 0.1  # Lowering nodes

        cr_mem[riv + clr] = cr_mem[riv] - clr * hnotch * dx  # Updating
        # remaining eroded volume
        cr_mem[riv:riv + clr] = 0

        return y, deb, riv + clr
    else:
        return y, deb, riv


# ********************************************************************************************

def RMSD(z1, z2):
    # Computes rms between 2 topographic profiles with same dx

    if len(z1) != len(z2):
        print('WTF!', len(z1), len(z2))

    rmsd = 0
    for i in range(0, len(z2)):
        rmsd = rmsd + (z2[i] - z1[i]) ** 2

    return (rmsd / len(z2)) ** 0.5


# ********************************************************************************************

def RMSD0(di, z1, z2):
    # Computes rms between 2 topographic profiles with different dx
    # x1 has the finest resolution

    n = 0
    rmsd = 0
    for i in range(0, len(z2) - 1):
        rmsd = rmsd + (z2[i] - z1[n * di]) ** 2
        n = n + 1

    return (rmsd / len(z2)) ** 0.5


# ********************************************************************************************

def Plot_Prof0(i, xz, prems, der, ow, dz, dt, dt_profile, tpt, ti, tmax, x, z,
               colors):
    # Plotting (or not) profile
    if dt < 1:
        if ti - tpt < dt:
            xz.plot(x, z + dz, color=colors[i], lw=0.5)
            tpt = tpt - dt_profile
    else:
        xz.plot(x, z + dz, color=colors[i], lw=0.5)

    # Plotting open water
    xz.plot(x[ow], z[ow] + dz, color='red', marker='.', ms=4)

    return tpt


# ********************************************************************************************

def setup_fig_Prof0(xz, imax, xmini, xmaxi, zmini, zmaxi, t, tit):
    fs = 12
    lwd = 1

    xz.tick_params(labelsize=fs)

    xz.set_xlabel('Distance (m)', fontsize=fs)
    xz.set_ylabel('Elevation (m)', fontsize=fs)
    xz.axis([xmini - 100, xmaxi + 100, zmini - 10, zmaxi + 10])
    xz.set_title(tit, ha='center', fontsize=fs)


# ********************************************************************************************
@jitmode
def rabout(prems, der, x, z):
    if prems <= 100:
        prems, der, x, z = raboutprems(prems, der, x, z)
    if der >= len(x) - 500:
        prems, der, x, z = raboutder(prems, der, x, z)

    return len(x), prems, der, x, z


# ********************************************************************************************
@jitmode
def adjust_limits(prems, der, xmini, xmaxi, zmini, zmaxi, dz, x, z):
    xmaxi = max(xmaxi, x[der])
    xmini = min(xmini, x[prems])
    zmaxi = max(zmaxi, z[der] + dz)
    zmini = min(zmini, z[prems] + dz)

    return xmini, xmaxi, zmini, zmaxi


# ********************************************************************************************
@jitmode
def u_story(imax, dt, u0):
    u = np.zeros(imax)
    u[:] = u0 * dt
    upmax = np.sum(u)

    return u, upmax


# ********************************************************************************************
@jitmode
def defdt(dt, zmin, G, dx, drsl_max, tmax, t):
    # Calculates dt and defines the number of iterations

    # If dt!=0, the value provided as input sets the time step, otherwise
    # dt is calculated here
    if dt == 0:

        # Vertical constraint
        if drsl_max > 0:
            dtmaxv = zmin / drsl_max
        else:
            dtmaxv = 1

        # Horizontal constraint
        if G != 0:
            dtmaxh = dx / G
        else:
            dtmaxh = 1

        dt = min(dtmaxv, dtmaxh)

    # Defining the number of iterations
    if tmax == -1:
        tmax = t[0]
    imax = int(tmax / dt)

    return imax, dt


# ********************************************************************************************

def disparam(slcurve, dt, dx, u, G, V0, slopi, tmax, tmaxb):
    # Displays simulation parameters

    print('******************************************************************')
    print('Vertical rate = ', u, 'mm/y')
    print('Reef growth rate = ', G, 'mm/y')
    print('Initial slope = ', slopi, '%')
    print('Eroded volume = ', V0, 'mm3/y')
    print('SL scenario = ', slcurve, ' over ', tmax, ' ky (', tmaxb, ')')
    print('Time step = ', dt, ' ky')
    print('Space step = ', dx, 'm')
    print('******************************************************************')


# ********************************************************************************************
@jitmode
def raboutprems(prems, der, x, y):
    # Adds nodes at the beginning of the profile in case it is too short
    # for the simulation

    xold = np.copy(x)
    yold = np.copy(y)
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    j0 = len(x)
    jmax = j0 + 5000
    x = np.zeros(jmax)
    y = np.zeros(jmax)

    for j in range(jmax - j0, jmax):
        x[j] = xold[j - jmax + j0]
        y[j] = yold[j - jmax + j0]
    for j in range(jmax - j0, -1, -1):
        x[j] = x[j + 1] - dx
        y[j] = y[j + 1] - dy

    return np.argmax(x > xold[prems]) - 1, np.argmax(x > xold[der]) - 1, \
           x, y


# ********************************************************************************************
@jitmode
def raboutder(prems, der, x, y):
    # Adds nodes to the profile in case it is too short for the simulation

    xold = np.copy(x)
    yold = np.copy(y)
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    j0 = len(x)
    jmax = j0 + 5000
    x = np.zeros(jmax)
    y = np.zeros(jmax)
    for j in range(0, j0):
        x[j] = xold[j]
        y[j] = yold[j]
    for j in range(j0, jmax):
        x[j] = x[j - 1] + dx
        y[j] = y[j - 1] + dy

    return np.argmax(x > xold[prems]) - 1, np.argmax(x > xold[der]) - 1, \
           x, y


# ********************************************************************************************
@jitmode
def geominidx(slopi, dx, upmax, emin):
    z_range = -emin + 1500 + abs(upmax)

    if upmax < 0:
        shiftz = -emin + 100
    else:
        shiftz = 200

    dz = dx * slopi / 100
    jmax = int(z_range / dz)

    x = np.arange(0., jmax + 1) * dx

    z = x * slopi / 100 - shiftz

    return x, z, jmax, x[-1], x[0], z[-1], z[0]


# ********************************************************************************************

def readfile(name):
    sl = open(name, "r")
    tmp = sl.read()
    col1 = np.genfromtxt(StringIO(tmp), dtype='float', usecols=(0))
    col2 = np.genfromtxt(StringIO(tmp), dtype='float', usecols=(1))
    sl.close()
    return col1, col2

# ********************************************************************************************
