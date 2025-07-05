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

    travelTime = 47.97593406427982 # days
    shipDeg = 60.71 # angle of ship at end

    earthRad = 150e6
    earthVel = (sun_mu/earthRad)**0.5
    marsRad = 228e6
    marsAngDisp = 360/687*travelTime # degrees
    marsDeg = shipDeg - marsAngDisp
    marsAng = marsDeg*math.pi/180
    marsVel = (sun_mu/marsRad)**0.5
    
    earthInitPos = np.array([earthRad, 0, 0])
    earthInitVel = np.array([0, earthVel, 0])
    marsInitPos = np.array([marsRad*math.cos(marsAng), marsRad*math.sin(marsAng), 0])
    marsInitVel = np.array([marsVel*math.cos(marsAng+math.pi/2), marsVel*math.sin(marsAng+math.pi/2), 0])

    integration_time = travelTime*86400+1
    integration_steps = 1000

    dry_mass = 100e3 # approximation in kg according to published interview with Elon Musk
    payload_mass = 150e3 # this and propellant mass found on SpaceX web page on Starship
    isp = 3350
    exhaust_v = 32.9 # km/s
    propellant_mass = 1500e3 # mass of all pulse units at start
    pulse_unit_mass = 79 # mass in kg of one pulse unit
    pulse_unit_number = propellant_mass // pulse_unit_mass + 1 # number of pulse units at start
    engine_mass = 107900
    raptor_mass = 1630 # mass of one raptor engine
    wet_mass = dry_mass + payload_mass + propellant_mass + engine_mass - raptor_mass * 6
    
    # Delta V of Hohmann transfer (for reference)
    #hohmannDeltaV1 = ((sun_mu/earthRad)**0.5) * ((2*marsRad/(earthRad+marsRad))**0.5 - 1) # delta v from departing burn (km/s)
    #print(hohmannDeltaV1)
    
    #propellant_1 = hohmann_mass * (1 - math.e**(-hohmannDeltaV1/(hohmann_isp*g))) # propellant expended by departing burn (kg) (same as Hohmann)
    
    propellant_1 = 11396.0*pulse_unit_mass # kg
    pulse_units_1 = propellant_1 // pulse_unit_mass + 1

    # Delta V of ship departing
    shipDeltaV1 = exhaust_v * math.log(wet_mass / (wet_mass - propellant_1)) # delta v from departing burn (km/s)
    shipInitVel = [0, earthVel+shipDeltaV1, 0]
    
    earth, times = keplerian_propagator(earthInitPos, earthInitVel, integration_time, integration_steps)
    mars, times = keplerian_propagator(marsInitPos, marsInitVel, integration_time, integration_steps)
    ship, times, arrival_time = ship_propagator(earthInitPos, shipInitVel, integration_time, integration_steps)

    # Delta V of ship arriving
    final_ship = ship[0:3,-1]
    final_x = final_ship[0]
    final_y = final_ship[1]
    unit_dir = [final_x/np.hypot(final_x,final_y), final_y/np.hypot(final_x,final_y)]
    mars_full, _ = keplerian_propagator(marsInitPos, marsInitVel, arrival_time, 10)
    mars_arrival_vel = mars_full[3:, -1]  # true velocity vector at arrival
    DV2_vector = ship[3:,-1] - mars_arrival_vel
    shipDeltaV2 = np.linalg.norm(DV2_vector)

    m_after_burn1 = wet_mass - propellant_1
    m_after_burn2 = m_after_burn1 / math.exp(shipDeltaV2 / (isp * g))
    propellant_2 = m_after_burn1 - m_after_burn2
    pulse_units_2 = propellant_2 // pulse_unit_mass + 1
    
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

    print(f"Ship angle from +X axis: {theta_deg:.2f} degrees")
    
    if arrival_time is not None:

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
        print("Transfer Time (days): "+str(arrival_time/86400))
        print("Delta V at Departure (km/s): "+str(shipDeltaV1))
        print("Delta V at Arrival (km/s): "+str(shipDeltaV2))
        print("Departing Pulse Unit Expenditure: "+str(pulse_units_1))
        print("Arriving Pulse Unit Expenditure: "+str(pulse_units_2))
        print("Total Pulse Unit Expenditure: "+str(pulse_units_1+pulse_units_2))
    
    plt.show()

def reach_mars_event(t, state):
    x, y, z = state[:3]
    r = np.linalg.norm([x, y, z])
    return r - 228e6

reach_mars_event.terminal = True
reach_mars_event.direction = 0 

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
    sol = solve_ivp(fun = lambda t,x:keplerian_eoms(t,x), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12, events=reach_mars_event)

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

if __name__ == '__main__':
    main()
