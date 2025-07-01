import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

def main():
    """
    Main function
    """
    
    # Gravitational Constant times Earth mass, adjusted for kilometers
    # earth_mu = 398600.441500000
    sun_mu = 1.989e30*6.67e-20 # * 1e-9 km^3/m^3

    dry_mass = 100e3 # approximation in kg according to published interview with Elon Musk
    payload_mass = 150e3 # this and propellant mass found on SpaceX web page on Starship
    propellant_mass = 110e3 # total capacity 1500e3
    reactor_mass = 20e3 # rough estimate
    wet_mass = dry_mass + payload_mass + propellant_mass + reactor_mass
    ship_mass = wet_mass

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

    shipInitVel = [0, earthVel, 0]
    shipInitState = np.concatenate((earthInitPos, shipInitVel, [ship_mass]))

    earth, times = keplerian_propagator(earthInitPos, earthInitVel, integration_time, integration_steps)
    mars, times = keplerian_propagator(marsInitPos, marsInitVel, integration_time, integration_steps)
    ship, times, arrival_time = ship_propagator(shipInitState, integration_time, integration_steps)                                   
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

    final_ship_mass = ship[-1, -1]
    if arrival_time is not None:
        print("Travel time (days): "+str(arrival_time/86400))
    print("Propellant Expended (kg): "+str(wet_mass - final_ship_mass))

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

def ship_propagator(init_state, tof, steps):
    """
    Function to propagate a given orbit
    """
    tspan = [0, tof]
    tof_array = np.linspace(0, tof, num=steps)
    sol = solve_ivp(fun=ship_eoms, t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol=1e-12, atol=1e-12, events=reach_mars_event)
    mars_arrival_time = sol.t_events[0][0] if sol.t_events[0].size > 0 else None
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

def ship_eoms(t, state):
    """
    Equation of motion for 2body orbits
    """
    
    # Extract values from init
    x, y, z, vx, vy, vz, ship_mass = state
    r_dot = np.array([vx, vy, vz])
    
    # Define r
    r = np.linalg.norm([x, y, z])

    sun_mu = 1.989e30*6.67e-20
    thrust_per_thruster = 2.08 # N
    thrusters = 6
    g = 9.80665
    speed = np.linalg.norm([vx,vy])

    thrust = thrust_per_thruster * thrusters # newtons
    acceleration = thrust/ship_mass*1e-3 # m/s^2 * 1e-3 km/m
    isp = 4163
    exhaust_v = isp * g # m/s
    mass_flow = thrust / exhaust_v
    
    # Solve for the acceleration
    ax = - (sun_mu/r**3) * x + vx/speed * acceleration
    ay = - (sun_mu/r**3) * y + vy/speed * acceleration
    az = - (sun_mu / r**3) * z

    dm = -mass_flow
    dx = np.array([vx, vy, vz, ax, ay, az, dm])

    return dx

if __name__ == '__main__':
    main()
