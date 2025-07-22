#! /bin/env python3
import math
import unittest
import numpy as np
import argparse
try:
    # sage library has integration functions that are more robust 
    # (in my experience), but it is not trivial to install
    from sage.all import *
    from sage.symbolic.integration.integral import definite_integral
    sagelib = True
except:
    from math import sin, cos, tan, atan, acos, asin, sqrt
    sagelib = False
    pass

import scipy.integrate as integrate
import scipy

base_l = 20     
min_b = 5
max_b = 61
base_b = 10
satellites = 75 # simulations use 75 sats in 6 orbits 30deg apart
min_beta = 8.2  # 8.2 deg is the minimum elevation 
min_beta_r = min_beta/360*2*math.pi
lux_lat = 49.6
lux_lat_r = lux_lat/360*2*math.pi
monte_carlo_calls = 50000

e = 6_378_000  # earth radius (m)
h = 780_000  # orbit height (m)
r = e + h
lambdA = e/r # alpha in Al Hourani's paper
phi_prime_h = math.pi/2 - min_beta_r



#
# test functions: they test some cases for which we can compute the 
# exact value of some function
# 
def compute_phi_max(b=base_b):
    """ in the idealized case of an infinite building, phip can never 
    be pi/2, because the building reaches the horizon, so we should never use
    an phip = pi/2. As a consequence, also phi has a limit. This function is 
    used only for testing and returns the maximum phi and phip"""
    phi_max = compute_phi(math.pi/2)
    WP_max = r*math.sin(phi_max)
    phip_max = math.pi/2 - math.atan(b/WP_max)
    return compute_phi(phip_max), phip_max

def compute_parallel(lat, l=0, b=0):
    """ lat,  l and b are just for compatibility with other approximations """
    return 2*math.pi*r*cos(lat)

class TestFormulas(unittest.TestCase):

    def test_phi(self):
        """ test corner cases of phi to phi'"""
        phi = 0.46
        phi_p = compute_phi_prime_robust(phi)
        self.assertLess(phi_p, math.pi/2)
        phi_r = compute_phi(phi_p)
        self.assertAlmostEqual(phi, phi_r)
        
    def test_alpha90(self):
        self.assertAlmostEqual(compute_alpha(0, 2, 1), math.pi/4)

    def test_WP0(self):
        wp = compute_WP(0)
        self.assertAlmostEqual(wp,h)

    def test_WP90(self):
        phi = compute_phi(math.pi/2)
        wp = compute_WP(phi)
        self.assertAlmostEqual(wp, (r**2-e**2)**0.5)

    def test_WQ_min(self):
        phi, _ = compute_phi_max()
        self.assertAlmostEqual(compute_WQ(phi, l=base_l, b=base_b), base_l/2, places=3)

    def test_WQ_max(self):
        phi = 0.00001
        self.assertAlmostEqual(compute_WQ(phi, l=base_l, b=base_b), h, places=2)

    def test_emisphere_satellite_number(self):
        theta_p = math.pi/4
        def sat_per_arc(phi, theta_p):
            return 1/r*sat_density(phi + theta_p)*compute_parallel(phi+theta_p)

        if sagelib:
            sats = monte_carlo_integral(sat_per_arc, [-math.pi/4], [math.pi/4], 
               params=(theta_p), calls=monte_carlo_calls)[0]
        else:
            sats = integrate.quad(sat_per_arc, -math.pi/4, math.pi/4,
                          args=(theta_p))[0]
        self.assertAlmostEqual(sats, satellites/2)

    def test_delta_0(self):
        phi_h = compute_phi(phi_prime_h)
        d = compute_delta(phi_h,0,)
        self.assertAlmostEqual(d, 0)

    def test_sky_view_area(self):
        # this computes the area of the spherical cap 
        # with radius r=1
        phi_h = compute_phi(phi_prime_h)
        theta_p = lux_lat_r # whatever
        area = compute_sats_in_sky_view(theta_p, s=0)
        cap_area = 2*math.pi*(1-cos(phi_h))
        self.assertAlmostEqual(area, cap_area, places=2)

    def test_arch0(self):
        phi = compute_phi(math.pi/2)
        a = compute_arch_exact(0, l=base_l, b=0.0001)
        self.assertAlmostEqual(a/(phi*2*r), 1, places=3)

    def test_arch90(self):
        phi, _ = compute_phi_max()
        self.assertAlmostEqual(compute_arch_exact(phi, l=base_l, b=base_b), 
                                base_l, places=3)

    def test_avg_lat(self):
        phi_p = 0.0001 #whatever
        phi = compute_phi(phi_p)
        omega = compute_omega(phi, base_l, base_b)[0]
        min_theta = compute_theta_phi_beta(omega, phi, lux_lat_r)
        max_theta = lux_lat_r-phi
        max_theta_computed = compute_theta_phi_beta(0, phi, lux_lat_r)
        avg_theta = compute_avg_lat_in_arch(phi, base_l, base_b, lux_lat_r)
        self.assertAlmostEqual(max_theta, max_theta_computed, places=5)
        self.assertGreater(max_theta, avg_theta)
        self.assertGreater(avg_theta, min_theta)


#
# some helper functions
#

def phi_boundaries(low_phi, high_phi):
    if not low_phi:
        low_phi = -compute_phi(phi_prime_h) 
    if not high_phi:
        high_phi = compute_phi(phi_prime_h) 
    return low_phi, high_phi

def compute_parallel_portion(phi, theta):
    d = compute_delta(phi, theta)
    arc = r*cos(phi+theta)*d
    return arc

#
# phi and phip conversion
#

def compute_phi(phi_p, e=e, h=h):
    """ this was computationally tested against AH formula, that is
    quite more complicate """
    return phi_p - asin(e/(e+h)*sin(phi_p))



def compute_phi_prime_robust(phi):
    """ The formula from AH paper has a singularity in pi/2, 
    this formula here is more robust """
    if phi < 0 :
        print('ERROR: phi is always positive when computing phi prime')
        exit()
    wp = sqrt(e**2 + (r)**2 - 2*e*r*cos(phi)) 
    eta = acos((wp**2 + r**2 - e**2)/(2*r*wp))
    phi_prime = math.pi - (math.pi - eta - phi)
    return phi_prime

#
# linear approximation functions: this was an initial attempt, now unised
#

def compute_alpha(phip, l=base_l, b=base_b):
    num = l*cos(phip)
    den = 2*b
    return atan(num/den)

def compute_WP(phi):
    return (r**2 + e**2 - 2*e*r*cos(phi))**0.5

def compute_WQ(phi, l=base_l, b=base_b):
    if phi < 0:
        phi = -phi
    phip = compute_phi_prime_robust(phi)
    WP = compute_WP(phi)
    return l*cos(phip)*WP/(2*b)

def compute_QT(phi, l=base_l, b=base_b):
    return 2*compute_WQ(phi, l, b)

#
# exact arc computation
#

def compute_omega(phi, l, b):
    if phi < 0 :
        phi = -phi # omega is symmetric over phi
    WP = compute_WP(phi)
    phip = compute_phi_prime_robust(phi)
    PCp = e*cos(phip)
    WCp = WP + PCp
    alpha = compute_alpha(phip, l, b)
    omega = alpha - asin(PCp/WCp*sin(alpha))
    return omega, WCp, WP, PCp, alpha

def compute_arch_exact(phi, l, b):
    omega, WCp, _, _, _ = compute_omega(phi, l, b)
    return WCp*omega*2

def compute_avg_lat_in_arch(phi, l, b, theta_p):

    omega, _, _, _, _ = compute_omega(phi,l,b)
    if sagelib:
        theta_phi = monte_carlo_integral(compute_theta_phi_beta, [0], [omega], 
            params=(phi, theta_p), calls=monte_carlo_calls)[0]
    else:
        theta_phi = integrate.quad(compute_theta_phi_beta, 0, omega,
                        args=(phi, theta_p))[0]
    return theta_phi/omega


def compute_theta_phi_beta(beta, phi, theta_p):
    WP = compute_WP(phi)
    if phi < 0 :
        phip = - compute_phi_prime_robust(-phi)
    else:
        phip = compute_phi_prime_robust(phi)

    PD = WP - (WP + e*cos(phip))*(1-cos(beta))
    Ds_z = e*sin(theta_p) + PD*sin(theta_p - phip)
    return asin(Ds_z/r)


 
#
# density functions
#

def sat_density(lat, s=satellites):
    if not s: # needed when we just compute an area
        return 1
    if lat > math.pi/2 and lat < math.pi:
        lat = math.pi - lat # flip on Y axis
    elif lat < -math.pi/2 and phi > math.pi:
        lat = lat - math.pi # flip on the origin

    return s/(2*math.pi**2*cos(lat))

#
# open sky view computation
#

def compute_delta(phi, theta):
    phi_h = compute_phi(phi_prime_h) 
    num = cos(phi_h)-sin(theta+phi)*sin(theta)
    den = cos(theta)*cos(theta+phi)
    return 2*acos(num/den)

def compute_sats_in_sky_view(theta, s=satellites, low_phi=0, high_phi=0):

    low_phi, high_phi = phi_boundaries(low_phi, high_phi)
    def sats_per_cap(phi, theta, s): 
        return 1/r*compute_parallel_portion(phi, theta)*sat_density(phi+theta, s=s)

    if sagelib:
        sats = monte_carlo_integral(sats_per_cap, [low_phi], [high_phi], 
           params=(theta,s), calls=monte_carlo_calls)[0]
    else:
        sats = integrate.quad(sats_per_cap, low_phi, high_phi,
                          args=(theta, s))[0]
    return sats

#
# canyon sky view computation
#

def sat_number(theta_p, l, b, approx='lin', low_phi=0, high_phi=0, axis='vertical'):
    """ here we assume theta is so that phi_h and -phi_h
    is in the same quadrant of theta """
    low_phi, high_phi = phi_boundaries(low_phi, high_phi)
    if approx == 'lin':
        func = compute_QT
    elif approx == 'half_circle':
        func = compute_parallel
    elif approx == 'exact':
        func = compute_arch_exact


    def sat_per_arc(phi, theta_p, l, b, axis, approx):
        if axis=='vertical':
            if approx == 'lin':
                lat = phi+theta_p
            elif approx == 'exact':
                lat = compute_avg_lat_in_arch(phi, l, b, theta_p) 
            return 1/r*sat_density(lat)*func(phi, l, b)
        else:
            return 1/r*sat_density(theta_p)*func(phi, l, b)

    if sagelib:
        sats = monte_carlo_integral(sat_per_arc, [low_phi], [high_phi], 
               params=(theta_p, l, b, axis, approx), calls=monte_carlo_calls)[0]
    else:
        sats = integrate.quad(sat_per_arc, low_phi, high_phi,
                         args=(theta_p, l, b, axis, approx))[0]
    
    return sats

#
# Performance functions
#

def compute_los(theta_p=lux_lat_r, l=base_l, b=base_b, axis='vertical', approx='exact'):
    sats = sat_number(theta_p, l, b, approx=approx, axis=axis)
    allsats = compute_sats_in_sky_view(theta_p, s=satellites)
    return sats/allsats

def compute_range_b(theta_p=lux_lat_r, l=base_l, axis='vertical', approx='exact'):
    for b in range(5,61):
        b = b
        sats = sat_number(theta_p, l, b, approx=approx, axis=axis)
        allsats = compute_sats_in_sky_view(theta_p, s=satellites)
        print(b, sats/allsats)

#
# Compare approximation functions
#

def compute_approximation_error(lmax=25, bmax=60, show=False):
    measure = {}
    
    for l in range(5, lmax+1, 5):
        measure[l] = {}
        for b in range(5, bmax+1, 5):
            #phi_min = compute_phi(phi_prime_h)
            low_phi, high_phi = phi_boundaries(0,0)

            values = []
            def avg_precision(phi, l, b):
                return (compute_QT(phi, l, b)-compute_arch_exact(phi, l, b))/compute_arch_exact(phi, l, b)
            
            if sagelib:
                area = monte_carlo_integral(avg_precision, [low_phi], [high_phi], 
                   params=(l, b), calls=monte_carlo_calls)[0]
            else:
                area = integrate.quad(avg_precision, low_phi, high_phi,
                         args=(l, b))[0]
            measure[l][b] = area/(high_phi-low_phi)
    plot_lines = {}
    for l in measure:
        plot_lines[l] = []
        for b in measure[l]:
            plot_lines[l].append([measure[l][b]])
            
    if show:
        import matplotlib.pyplot as plt
        for l in measure:
            plt.plot(measure[l].keys(), plot_lines[l], label=f'street width: {l}')
        plt.legend()
        plt.title(f'Approximation error h={h/1000}km')
        plt.xlabel('Buildings heigh')
        plt.ylabel('Relative error lin/exact')
        plt.show()
    else:
        print('canyon_width', ' '.join([str(x) for x in measure.keys()]))
        for b in measure[list(measure.keys())[0]]:
            print(b, ' '.join([str(np.average(measure[i][b])) for i in measure.keys()]))
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=int, default=20, help='street width')
    parser.add_argument('-v', default=False, help='vertical canyon (default is horizontal)', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.v:
        axis='vertical'
    else:
        axis='horizontal'
    print(f'Plotting P_LoS for a {axis} canyon as a function of the building heigth, for street width set to {args.l}\n')
    compute_range_b(l=args.l, axis=axis, approx='exact')
