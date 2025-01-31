import numpy as np
from io import StringIO
import math

#from .context import jitmode


# *****************************************************************************
#@jitmode
def defdt(zmin, G, dx, drsl_max):
    # Vertical constraint
    if drsl_max > 0:
        dtmaxv = zmin / drsl_max
    else:
        dtmaxv = 10000

    # Horizontal constraint
    dtmaxh = dx / G

    return min(dtmaxv, dtmaxh)

# *****************************************************************************
def u_story(imax, dt, u0):

    u=np.zeros(imax)
    u[:]=u0*dt
    upmax=np.sum(u)

    return(u, upmax)

#
# *****************************************************************************

def figparam(dtx, xt, yt, ax, slcurve, u, G, slopi, zmax, dt, dx):
    # Writes simulation parameters on figure

    if dtx < 30:
        dtx = 30

    fts = 4
    p = "SL curve = " + slcurve
    ax.text(xt, yt, p, fontsize=fts)
    p = "Steps = " + str("%.2f" % dt) + ' ky - ' + str(dx) + ' m'
    yt = yt - dtx
    ax.text(xt, yt, p, fontsize=fts)
    p = "U = " + str(u) + " (" + str(u * dt) + ") mm/y"
    yt = yt - dtx
    ax.text(xt, yt, p, fontsize=fts)
    p = "G = " + str(G) + " (" + str(G * dt) + ") mm/y"
    yt = yt - dtx
    ax.text(xt, yt, p, fontsize=fts)
    p = "Slope = " + str(slopi) + "%"
    yt = yt - dtx
    ax.text(xt, yt, p, fontsize=fts)
    p = "Zmax = " + str(zmax) + " m"
    yt = yt - dtx
    ax.text(xt, yt, p, fontsize=fts)


# *****************************************************************************

def disparam(slcurve, dt, dx, u, G, slopi, tmax, V0):
    # Displays simulation parameters

    print('******************************************************************')
    print('Vertical rate = ', u, 'mm/y')
    print('Reef growth rate = ', G, 'mm/y')
    print('Initial slope = ', slopi, '%')
    print('Eroded volume = ', V0, 'mm3/y')
    print('SL scenario = ', slcurve, ' over ', tmax, ' ky')
    print('Time step = ', dt, ' ky')
    print('Space step = ', dx, 'm')
    print('******************************************************************')


# *****************************************************************************

#@jitmode
def raboutprems(prems, der, x, y):
    # Adds nodes at the beginning of the profile in case it is too short for
    # the simulation

    xold = np.copy(x)
    yold = np.copy(y)
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    j0 = len(x)
    jmax = j0 + 5000
    x = np.zeros(jmax)
    y = np.zeros(jmax)

    for j in range(jmax - j0, jmax):
        #        print('jmax-j0', jmax-j0, j, jmax)
        x[j] = xold[j - jmax + j0]
        y[j] = yold[j - jmax + j0]
    for j in range(jmax - j0, -1, -1):
        x[j] = x[j + 1] - dx
        y[j] = y[j + 1] - dy
    return len(x), np.argmax(x > xold[prems]) - 1, np.argmax(
        x > xold[der]) - 1, x, y


# *****************************************************************************

#@jitmode
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

    return len(x), np.argmax(x > xold[prems]) - 1, np.argmax(
        x > xold[der]) - 1, x, y


# *****************************************************************************

#@jitmode
def geominidx(slopi, dx, upmax, emin):
    #   print('jmax', jmax, 'len', len(xini))
    #    shifty=xini[jmax]*slopi/200

    ymax = 200

    if upmax > 0:
        shifty = -emin + upmax + ymax
    #        print('shifty', shifty)
    else:
        shifty = -emin + ymax

    dy = dx * slopi / 100
    jmax = int((ymax + shifty) / dy) + 1

    x = np.arange(0., jmax + 1) * dx

    y = x * slopi / 100
    y = y - shifty

    return x, y, jmax, x[-1], x[0], y[-1], y[0]


# *****************************************************************************

def readfile(name):
    sl = open(name, "r")
    tmp = sl.read()
    col1 = np.genfromtxt(StringIO(tmp), dtype='float', usecols=(0,))
    col2 = np.genfromtxt(StringIO(tmp), dtype='float', usecols=(1,))
    sl.close()
    return col1, col2


# *****************************************************************************

#@jitmode
def resampdx(dbg, deb, fin, dx, xin, yin):
    # Resamples the profile after construction along regular dx

    k = deb - 1
    xout = np.copy(xin)
    jmax = len(xin)
    yout = np.copy(yin)

    for j in range(deb - 1, fin + 1):
        xout[j] = xout[j - 1] + dx
        if xin[k] == xout[j]:
            yout[j] = yin[k]
            k = k + 1

        else:
            n = 0
            while xin[k + n] < xout[j]:
                n = n + 1

            if n > 0:
                yout[j] = linterpy2(xout[j], xin[k + n - 1], xin[k + n],
                                    yin[k + n - 1], yin[k + n])
                k = k + n
                # if n>2:
                #   print('Must be a couille in the potage', n, j, xin[k-1],
                #   xin[k], xout[j])
                #  sys.exit()
            else:
                yout[j] = linterpy2(xout[j], xin[k - 1], xin[k], yin[k - 1],
                                    yin[k])

    return xout, yout


# *****************************************************************************

#@jitmode
def linterpy2(xf, x1, x2, y1, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a * xf + b

# *****************************************************************************
def defdt_eros(dt, V0, beta, tmax, t):
# Calculates dt and defines the number of iterations

    # If dt!=0, the value provided as input sets the time step, otherwise dt is calculated here
    if dt==0:

        dt=0.1/(V0*math.exp(-4)*beta)

    #Defining the number of iterations
    if tmax==-1:                                                # Simulation through the whole SL curve
        tmax=t[0]
    imax=int(tmax/dt)

    return imax, dt

# *****************************************************************************
def erosion(u, V, zb, beta1, e, dx, y, cr_mem):
# Based on Anderson et al., 1999

    beta2=1                                                     # Coefficient for erosion efficiency, cliff retreat
    hnotch=1                                                    # Height of notch for volume eroded during cliff retreat
    Vtmp=0

    # Variables initialisation
    deb=np.argmax(e-y<=zb)                                      # First node for wave erosion
    riv=np.argmax(e<=y)                                         # Last node for sea bed erosion
    Vrest=V

    # Vertical sea-bed erosion
    for j in range(deb, riv):
        dh=e-y[j]                                               # Water height
        if dh<=zb and Vrest >0:
            dV=min(Vrest*math.exp(-dh/(zb/4))*beta1, Vrest/dx)
            y[j]=y[j]-dV
            Vrest=Vrest-dV*dx

    # Cliff retreat
    cr_mem[riv]=cr_mem[riv]+Vrest

    if cr_mem[riv]>=dx*hnotch:                                  # Enough eroded volume at this node for cliff retreat
        clr=int(cr_mem[riv]/(hnotch*beta2*dx))                  # Number of nodes exposed to cliff retreat
        cr_mem[riv]=cr_mem[riv]+np.sum(cr_mem[riv+1:riv+clr+1])

        while cr_mem[riv]-clr*hnotch*dx > hnotch*dx:
            #print('Ca depasse !!!')
            clr=clr+1
            Vtmp=Vtmp+cr_mem[clr]
        y[riv:riv+clr]=e-0.1                                    # Lowering nodes

        cr_mem[riv+clr]=cr_mem[riv]-clr*hnotch*dx               # Updating remaining eroded volume
        cr_mem[riv:riv+clr]=0


        return y, deb, riv+clr
    else:
        return y, deb, riv

#********************************************************************************************
def rabout(prems, der, x, z):

    if prems<=100:
        #print('rabout prems')
        prems, der, x, z = raboutprems(prems, der, x, z)
    if der>=len(x)-500:
        #print('rabout der')
        prems, der, x, z = raboutder(prems, der, x, z)

    return len(x), prems, der, x, z

#********************************************************************************************