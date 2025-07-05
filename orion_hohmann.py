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

    travelTime = 259.38719214052634 # days
    shipDeg = 180 # angle of ship at end

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

    # Delta V of ship (Hohmann)
    shipDeltaV1 = ((sun_mu/earthRad)**0.5) * ((2*marsRad/(earthRad+marsRad))**0.5 - 1) # delta v from departing burn (km/s)
    shipDeltaV2 = ((sun_mu/marsRad)**0.5) * (1 - (2*earthRad/(earthRad+marsRad))**0.5) # delta v from arriving burn (km/s)
    shipInitVel = [0, earthVel+shipDeltaV1, 0]

    dry_mass = 100e3 # approximation in kg according to published interview with Elon Musk
    payload_mass = 150e3 # this and propellant mass found on SpaceX web page on Starship
    isp = 3350
    exhaust_v = 32.9 # km/s
    pulse_unit_mass = 79 # mass in kg of one pulse unit
    pulse_unit_number = 814 # number of pulse units at start
    propellant_mass = pulse_unit_mass * pulse_unit_number # mass of all pulse units at start
    engine_mass = 107900
    raptor_mass = 1630 # mass of one raptor engine
    wet_mass = dry_mass + payload_mass + propellant_mass + engine_mass - raptor_mass * 6
    
    pulse_units_1 = (wet_mass * (1 - math.e**(-shipDeltaV1/(isp*g)))) // pulse_unit_mass + 1 # number of pulse units expended by departing burn
    pulse_units_2 = ((wet_mass - pulse_units_1*pulse_unit_mass) * (1 - math.e**(-shipDeltaV2/(isp*g)))) // pulse_unit_mass + 1 # number of pulse units expended by arriving burn
    pulse_units_total = pulse_units_1 + pulse_units_2

    earth, times = keplerian_propagator(earthInitPos, earthInitVel, integration_time, integration_steps)
    mars, times = keplerian_propagator(marsInitPos, marsInitVel, integration_time, integration_steps)
    ship, times = keplerian_propagator(earthInitPos, shipInitVel, integration_time, integration_steps)
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

    print("Transfer Time (days): "+str(integration_time/86400))
    print("Delta V at Departure (km/s): "+str(shipDeltaV1))
    print("Delta V at Arrival (km/s): "+str(shipDeltaV2))
    print("Departing Pulse Unit Expenditure: "+str(pulse_units_1))
    print("Arriving Pulse Unit Expenditure: "+str(pulse_units_2))
    print("Total Pulse Unit Expenditure: "+str(pulse_units_total))
    
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
