import numpy as np
import math
import sys

from .context import jitmode


# *****************************************************************************

@jitmode
def erosion(u, v0, zb, edeb, efin, dx, y):
    # Based on Anderson et al., 1999

    eps = 1e-280  # Threshold value for the end of erosion
    beta1 = 0.1  # Coefficient for erosion efficiency, sea bed abrasion
    beta2 = 1  # Coefficient for erosion efficiency, cliff retreat
    hnotch = 1  # Height for volume eroded during cliff retreat

    # Defining the number of iterations, one for each meter of RSL variation
    n = abs(int(round(efin - edeb - u)))

    if n == 0 or n == 1:
        n = 1
    else:
        n = n - 1
    de = (edeb - efin) / n
    e = edeb
    V = v0 / n

    for i in range(n):
        e = e + de
        deb = np.argmax(e - y <= zb)
        riv = np.argmax(e < y) - 1
        Vrest = V

        # Vertical sea-bed erosion
        for j in range(deb, riv):
            dh = e - y[j]
            if Vrest > eps and dh <= zb:
                dV = Vrest * math.exp(-dh / (zb / 4)) * beta1
                y[j] = y[j] - dV
                Vrest = Vrest - dV * dx

        # Cliff retreat
        # depx=Vrest/hnotch*beta2
        # y[riv+1:riv+1+int(depx)]=y[riv]
        y[riv + 1:riv + 1 + int(Vrest)] = y[riv]

    return y, deb, riv + 1 + int(Vrest)


# *****************************************************************************

@jitmode
def const(dbg, zow, G, zmin, zreef, eu, x, y):
    a = 15
    dj = 1
    slp = np.zeros(len(x))

    # Computing water height
    dh = eu - y

    # Narrowing around zone of reef growth
    deb = np.argmax(zreef >= dh)
    fin = np.argmax(eu <= y)

    if dh[deb] < 0:
        # print('No construction')
        return x, y, deb - 1, fin

    # Searching for xow for horizontal gradient
    ow = np.argmax(zow >= dh)

    # Computing slope
    for j in range(dj, len(x) - dj):
        if x[j + dj] == x[j - dj]:
            print('PROBLEME reef growth', j, x[j - dj], x[j + dj])
            sys.exit()
        slp[j] = np.arctan((y[j + dj] - y[j - dj]) / (x[j + dj] - x[j - dj]))

    # Constructing
    for j in range(deb, fin + 1):
        if zreef >= dh[j] > 0.1:

            # Vertical gradient
            if dh[j] <= zmin:
                dep1 = (1. + np.cos(np.pi * dh[j] / zmin - np.pi)) / 2
            else:
                dep1 = (1. + np.cos(np.pi * dh[j] / zreef)) / 2

            # Horizontal gradient
            dep2 = np.tanh((x[ow] - x[j]) / a) / 2 + 0.5

            dep = dep1 * G * dep2

            depx = -dep * np.sin(slp[j])
            if (depx > 0 and slp[j] > 0) or (depx > 0 and slp[j] > 0) or (
                    slp[j] == 0 and depx != 0):
                # [NH] remove i from print
                print("PB", j, x[j], 'depx', depx, 'pente', slp[j],
                      np.sin(slp[j]), 'dep1', dep1)
                sys.exit()
            depy = dep * np.cos(slp[j])

            # Limitation by SL and inner reef
            depy = min(depy, dh[j] - 0.1)
            depy = max(0., depy)

            y[j] = y[j] + depy
            x[j] = x[j] + depx

    return x, y, deb, fin

# *****************************************************************************
