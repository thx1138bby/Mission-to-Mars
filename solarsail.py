import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

arrived = False

def main():
    """
    Main function
    """
    
    # Gravitational Constant times Earth mass, adjusted for kilometers
    # earth_mu = 398600.441500000
    sun_mu = 1.989e30*6.67e-20 # * 1e-9 km^3/m^3
    g = 9.80665*1e-3 # km/s^2

    earthRad = 150e6
    earthVel = (sun_mu/earthRad)**0.5
    marsRad = 228e6
    marsVel = (sun_mu/marsRad)**0.5
    
    earthInitPos = np.array([earthRad,0,0])
    earthInitVel = np.array([0,earthVel, 0])
    marsInitPos = np.array([0, marsRad, 0])
    marsInitVel = np.array([-marsVel, 0, 0])

    integration_time = 24*60*60*28*6.54
    integration_steps = 1000

    # Delta V of ship (Hohmann)
    shipDeltaV1 = ((sun_mu/earthRad)**0.5) * ((2*marsRad/(earthRad+marsRad))**0.5 - 1) # delta v from departing burn (km/s)
    shipDeltaV2 = ((sun_mu/marsRad)**0.5) * (1 - (2*earthRad/(earthRad+marsRad))**0.5) # delta v from arriving burn (km/s)
    shipInitVel = [0, earthVel+shipDeltaV1, 0]

    earth, times = keplerian_propagator(earthInitPos, earthInitVel, integration_time, integration_steps)
    mars, times = keplerian_propagator(marsInitPos, marsInitVel, integration_time, integration_steps)
    ship, times = ship_propagator(earthInitPos, shipInitVel, integration_time, integration_steps)
    # Plot it
    fig = plt.figure()
    # Define axes in that figure
    ax = plt.axes(projection='3d',computed_zorder=False)
    # Plot x, y, z
    ax.plot(earth[0],earth[1],earth[2],zorder=5)
    ax.plot(mars[0],mars[1],mars[2],zorder=5)
    ax.plot(ship[0],ship[1],ship[2],zorder=5)
    plt.title("All Orbits")
    ax.set_xlabel("X-axis (km)")
    ax.set_ylabel("Y-axis (km)")
    ax.set_zlabel("Z-axis (km)")
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.zaxis.set_tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='box')

    if arrived:
        print(arrived)

    print(integration_time/86400)

    plt.show()
    

def keplerian_propagator(init_r, init_v, tof, steps):
    """
    Function to propagate a given orbit
    """
    # Time vector
    tspan = [0, tof]
    # Array of time values
    tof_array = np.linspace(0,tof, num=steps)
    init_state = np.concatenate((init_r,init_v))
    # Do the integration
    sol = solve_ivp(fun = lambda t,x:keplerian_eoms(t,x), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)

    # Return everything
    return sol.y, sol.t

def ship_propagator(init_r, init_v, tof, steps):
    """
    Function to propagate a given orbit
    """
    # Time vector
    tspan = [0, tof]
    # Array of time values
    tof_array = np.linspace(0,tof, num=steps)
    init_state = np.concatenate((init_r,init_v))
    # Do the integration
    sol = solve_ivp(fun = lambda t,x:ship_eoms(t,x), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)

    # Return everything
    return sol.y, sol.t


def keplerian_eoms(t, state):
    """
    Equation of motion for 2body orbits
    """
    sun_mu = 1.989e30*6.67e-20
    
    # Extract values from init
    x, y, z, vx, vy, vz = state
    r_dot = np.array([vx, vy, vz])
    
    # Define r
    r = np.linalg.norm([x, y, z])
    # Solve for the acceleration
    ax = - (sun_mu/r**3) * x
    ay = - (sun_mu/r**3) * y
    az = - (sun_mu/r**3) * z

    v_dot = np.array([ax, ay, az])

    dx = np.append(r_dot, v_dot)

    return dx

def ship_eoms(t, state):
    """
    Equation of motion for 2body orbits
    """
    
    # Extract values from init
    x, y, z, vx, vy, vz = state
    r_dot = np.array([vx, vy, vz])
    
    # Define r
    r = np.linalg.norm([x, y, z])
    
    sun_mu = 1.989e30*6.67e-20
    solar_constant = 1.361e6 # kw/m^2 * 1e6 m^2/km^2
    au = 150e6 # km
    lightspeed = 299792 # km/s
    sail_width = 2 # km
    sail_area = sail_width**2 # km^2
    rad_pressure = solar_constant/(lightspeed*(r/au)**2) # Pa
    force = sail_area*rad_pressure
    dry_mass = 100e3
    
    acceleration = force/dry_mass*1e-3 # m/s^2 * 1e-3 km/m
    
    # Solve for the acceleration
    ax = - (sun_mu/r**3) * x + x/r * acceleration
    ay = - (sun_mu/r**3) * y + y/r * acceleration
    az = - (sun_mu/r**3) * z

    v_dot = np.array([ax, ay, az])

    dx = np.append(r_dot, v_dot)

    global arrived
    if r >= 228e6:
        arrived = True

    return dx


if __name__ == '__main__':
    main()
