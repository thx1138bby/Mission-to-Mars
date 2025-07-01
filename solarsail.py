import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial
import math

mass = 1
area = 1

def main():
    """
    Main function
    """
    
    # Gravitational Constant times Earth mass, adjusted for kilometers
    # earth_mu = 398600.441500000
    sun_mu = 1.989e30*6.67e-20 # * 1e-9 km^3/m^3
    g = 9.80665*1e-3 # km/s^2
    isp = 841

    earthRad = 150e6
    earthVel = (sun_mu/earthRad)**0.5
    marsRad = 228e6
    marsVel = (sun_mu/marsRad)**0.5
    
    earthInitPos = np.array([earthRad, 0, 0])
    earthInitVel = np.array([0, earthVel, 0])
    marsInitPos = np.array([marsRad, 0, 0])
    marsInitVel = np.array([0, marsVel, 0])

    integration_time = 24*60*60*365*3 # three years, arbitrarily high to ensure no cutoff
    integration_steps = 1000

    sail_width = 1000 # m
    sail_area = sail_width**2 # m^2
    sail_density = 1400 # kg/m^3
    sail_thickness = 2.5 * 1e-6 # 2.5 um to m
    dry_mass = 100e3
    payload_mass = 150e3
    propellant_mass = 1500e3
    sail_mass = sail_area * sail_thickness * sail_density
    wet_mass = dry_mass + payload_mass + sail_mass + propellant_mass

    shipDeltaV1 = ((sun_mu/earthRad)**0.5) * ((2*marsRad/(earthRad+marsRad))**0.5 - 1) # delta v from departing burn (km/s)
    shipInitVel = [0, earthVel+shipDeltaV1, 0]

    earth, times = keplerian_propagator(earthInitPos, earthInitVel, integration_time, integration_steps)
    mars, times = keplerian_propagator(marsInitPos, marsInitVel, integration_time, integration_steps)
    ship, times, arrival_time = ship_propagator(earthInitPos, shipInitVel, integration_time, integration_steps, wet_mass, sail_area)
    
    # Final ship velocity ship[3:,-1]
    final_ship = ship[0:3,-1]
    final_x = final_ship[0]
    final_y = final_ship[1]
    unit_dir = [final_x/np.hypot(final_x,final_y), final_y/np.hypot(final_x,final_y)]
    mars_time_index = np.searchsorted(times, arrival_time)
    mars_vel_vector = mars[3:6, mars_time_index]
    DV2_vector = ship[3:,-1] - mars_vel_vector
    shipDeltaV2 = np.linalg.norm(DV2_vector)
    
    propellant_1 = wet_mass * (1 - math.e**(-shipDeltaV1/(isp*g))) # propellant expended by departing burn (kg)
    propellant_2 = (wet_mass - propellant_1) * (1 - math.exp(-shipDeltaV2 / (isp * g)))

    global mass
    global area

    mass = wet_mass - propellant_1
    area = sail_area
    
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

    if arrival_time is not None:
        print("Travel time (days): "+str(arrival_time/86400))
        print("Delta V departing (km/s): "+str(shipDeltaV1))
        print("Propellant expenditure arriving (t): "+str(propellant_1*1e-3))
        print("Delta V arriving (km/s): "+str(shipDeltaV2))
        print("Propellant expenditure arriving (t): "+str(propellant_2*1e-3))
        print("Total propellant expenditure (t): "+str((propellant_1 + propellant_2)*1e-3))

    plt.show()

def reach_mars_event(t, state):
    x, y, z = state[:3]
    r = np.linalg.norm([x, y, z])
    return r - 228e6

reach_mars_event.terminal = True
reach_mars_event.direction = 1 

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

def ship_propagator(init_r, init_v, tof, steps, ship_mass, sail_area):
    """
    Function to propagate a given orbit
    """
    # Time vector
    tspan = [0, tof]
    # Array of time values
    tof_array = np.linspace(0,tof, num=steps)
    init_state = np.concatenate((init_r,init_v))
    ship_dynamics = partial(ship_eoms, ship_mass=ship_mass, sail_area=sail_area)
    # Do the integration
    sol = solve_ivp(fun = ship_dynamics, t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12, events=reach_mars_event)

    mars_arrival_time = sol.t_events[0][0] if sol.t_events[0].size > 0 else None

    # Return everything
    return sol.y, sol.t, mars_arrival_time

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

def ship_eoms(t, state, ship_mass, sail_area):
    """
    Equation of motion for 2body orbits
    """
    
    # Extract values from init
    x, y, z, vx, vy, vz = state
    r_dot = np.array([vx, vy, vz])
    
    # Define r
    r = np.linalg.norm([x, y, z])
    
    sun_mu = 1.989e30*6.67e-20
    solar_constant = 1361 # w/m^2
    au = 150e6 # km
    lightspeed = 299792458 # m/s
    rad_pressure = 2 * solar_constant / (lightspeed*(r/au)**2) # Pa
    ship_mass = mass
    sail_area = area
    force = sail_area*rad_pressure
    
    acceleration = force/ship_mass*1e-3 # m/s^2 * 1e-3 km/m
    # Solve for the acceleration
    ax = - (sun_mu/r**3) * x + x/r * acceleration
    ay = - (sun_mu/r**3) * y + y/r * acceleration
    az = - (sun_mu/r**3) * z

    v_dot = np.array([ax, ay, az])

    dx = np.append(r_dot, v_dot)

    return dx


if __name__ == '__main__':
    main()
