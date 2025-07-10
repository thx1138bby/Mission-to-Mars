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
    g = 9.80665*1e-3 # km/s^2

    dry_mass = 100e3 # approximation in kg according to published interview with Elon Musk
    payload_mass = 150e3 # this and propellant mass found on SpaceX web page on Starship
    propellant_mass = 1193e3 # total capacity 1500e3
    engine_mass = 18144 # mass of nerva engine (kg)
    wet_mass = dry_mass + payload_mass + propellant_mass + engine_mass
    nervaIsp = 841

    earthRad = 150e6
    earthVel = (sun_mu/earthRad)**0.5
    marsRad = 228e6
    marsVel = (sun_mu/marsRad)**0.5
    
    earthInitPos = np.array([earthRad, 0, 0])
    earthInitVel = np.array([0, earthVel, 0])
    marsInitPos = np.array([marsRad, 0, 0])
    marsInitVel = np.array([0, marsVel, 0])

    integration_time = 365*24*60*60
    integration_steps = 1000

    shipDeltaV1 = ((sun_mu/earthRad)**0.5) * ((2*marsRad/(earthRad+marsRad))**0.5 - 1) # delta v from departing burn (km/s)
    shipInitVel = [0, earthVel+shipDeltaV1, 0]
    propellant_1 = wet_mass * (1 - math.e**(-shipDeltaV1/(nervaIsp*g))) # nerva propellant expended by departing burn (kg)

    ship_mass = wet_mass - propellant_1
    shipInitState = np.concatenate((earthInitPos, shipInitVel, [ship_mass]))

    ship, ship_times, arrival_time = ship_propagator(shipInitState, integration_time, integration_steps)
    mars_ToF, angular_velocity = calculate_mars_angle(ship, marsRad, sun_mu)
    earth, times = keplerian_propagator(earthInitPos, earthInitVel, arrival_time, integration_steps)
    mars, times = keplerian_propagator(marsInitPos, marsInitVel, mars_ToF, integration_steps)

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

    # Final ship velocity ship[3:,-1]
    final_ship = ship[0:3,-1]
    final_x = final_ship[0]
    final_y = final_ship[1]

    # Angle from +X axis (in radians)
    theta_rad = np.arctan2(final_y, final_x)

    # Convert to degrees
    theta_deg = np.degrees(theta_rad)

    # Ensure it's in [0, 360)
    if theta_deg < 0:
        theta_deg += 360

    print("Ship angle from +X axis (degrees): "+str(theta_deg))

    # Get final positions
    final_ship_pos = ship[0:3, -1]  # X, Y, Z of Earth at final time
    final_mars_pos = mars[0:3, -1]    # X, Y, Z of Mars at final time

    final_mars_x = final_mars_pos[0]
    final_mars_y = final_mars_pos[1]
    mars_theta_rad = np.arctan2(final_mars_y, final_mars_x)
    mars_theta_deg = np.degrees(mars_theta_rad)
    print("Ship - Mars Angle (deg): "+str(theta_deg - mars_theta_deg))

    # Compute distance in km
    distance = np.linalg.norm(final_mars_pos - final_ship_pos)
    print("Ship to Mars Distance (km): "+str(distance))
    
    DV2_vector = [mars[3][-1]-ship[3][-1], mars[4][-1]-ship[4][-1]]
    shipDeltaV2 = np.linalg.norm(DV2_vector)
    final_mass = ship[6, -1]
    propellant_2 = final_mass * (1 - math.exp(-shipDeltaV2 / (nervaIsp * g))) # nerva propellant

    exhaust_v = calculate_ship(final_mass)[2]*1e-3
    plasmaDeltaV = exhaust_v * math.log(ship_mass/final_mass)
    if arrival_time is not None:
        print("Travel time (days): "+str(arrival_time/86400))
    print("NTR Delta V Departing (km/s): "+str(shipDeltaV1))
    print("Plasma Delta V (km/s): "+str(plasmaDeltaV))
    print("NTR Delta V Arriving (km/s): "+str(shipDeltaV2))
    print("NTR Propellant Expended Departing (t): "+str(propellant_1*1e-3))
    print("Plasma Propellant Expended (t): "+str((ship_mass - final_mass)*1e-3))
    print("NTR Propellant Expended Arriving (t): "+str(propellant_2*1e-3))
    print("Total Propellant Expended (t): "+str((propellant_1 + ship_mass - final_mass + propellant_2)*1e-3))

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
    sol = solve_ivp(fun = lambda t,x:keplerian_eoms(t,x), t_span=tspan, y0=init_state, method="DOP853", rtol = 1e-12, atol = 1e-12)

    # Return everything
    return sol.y, sol.t

def ship_propagator(init_state, tof, steps):
    """
    Function to propagate a given orbit
    """
    tspan = [0, tof]
    tof_array = np.linspace(0, tof, num=steps)
    sol = solve_ivp(fun=ship_eoms, t_span=tspan, y0=init_state, method="DOP853", rtol=1e-12, atol=1e-12, events=reach_mars_event)
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

def calculate_ship(ship_mass):
    rAnode = 0.026
    rCathode = 0.0127
    field = 0.3 # T
    current = 1500 # A
    thrust_per_thruster = 1/2**0.5*field*current*rAnode*(1 - 3/2*(rCathode/rAnode)**2)
    thrusters = 40
    g = 9.80665

    thrust = thrust_per_thruster * thrusters # newtons
    acceleration = thrust/ship_mass*1e-3 # m/s^2 * 1e-3 km/m
    isp = 1100
    exhaust_v = isp * g # m/s
    mass_flow = thrust / exhaust_v

    return acceleration, mass_flow, exhaust_v

def ship_eoms(t, state):
    """
    Equation of motion for 2body orbits
    """
    
    # Extract values from init
    x, y, z, vx, vy, vz, ship_mass = state
    r_dot = np.array([vx, vy, vz])

    sun_mu = 1.989e30*6.67e-20
    
    # Define r
    r = np.linalg.norm([x, y, z])

    dry_mass = 100e3 + 150e3 + 18144

    dm = 0

    # Solve for the acceleration
    ax = - (sun_mu/r**3) * x 
    ay = - (sun_mu/r**3) * y
    az = - (sun_mu / r**3) * z
    if ship_mass > dry_mass:
        acceleration = calculate_ship(ship_mass)[0]
        mass_flow = calculate_ship(ship_mass)[1]
        speed = np.linalg.norm([vx,vy])
        ax += vx/speed * acceleration
        ay += vy/speed * acceleration
        dm = -mass_flow
    
    dx = np.array([vx, vy, vz, ax, ay, az, dm])

    return dx

def calculate_mars_angle(ship_traj, marsRad, sun_mu):
    """
    Function to calculate the init angle of mars to accomplish rendezvous
    """
    final_x = ship_traj[0][-1]
    final_y = ship_traj[1][-1]

    # Angle from +X axis (in radians)
    theta_rad = np.arctan2(final_y, final_x)

    # Ensure it's in [0, 360)
    if theta_rad < 0:
        theta_rad += 2*np.pi

    # What is the Time of Flight for Mars
    # To accomplish this angle
    period = 2*np.pi*np.sqrt(marsRad**3/sun_mu) 

    angular_velocity = (2*np.pi)/period #radians/second
    # What time offset accomplishes this angular offset
    time_offset = theta_rad/angular_velocity

    return time_offset, angular_velocity

if __name__ == '__main__':
    main()
