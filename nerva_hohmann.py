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

    earthRad = 150e6
    earthVel = (sun_mu/earthRad)**0.5
    marsRad = 228e6
    marsVel = (sun_mu/marsRad)**0.5
    
    earthInitPos = np.array([earthRad, 0, 0])
    earthInitVel = np.array([0, earthVel, 0])
    marsInitPos = np.array([marsRad, 0, 0])
    marsInitVel = np.array([0, marsVel, 0])

    integration_time = (2*math.pi/((sun_mu)**0.5)*(((earthRad+marsRad)/2)**1.5))/2
    integration_steps = 1000

    # Delta V of ship (Hohmann)
    shipDeltaV1 = ((sun_mu/earthRad)**0.5) * ((2*marsRad/(earthRad+marsRad))**0.5 - 1) # delta v from departing burn (km/s)
    shipDeltaV2 = ((sun_mu/marsRad)**0.5) * (1 - (2*earthRad/(earthRad+marsRad))**0.5) # delta v from arriving burn (km/s)
    shipInitVel = [0, earthVel+shipDeltaV1, 0]

    dry_mass = 100e3 # approximation in kg according to published interview with Elon Musk
    payload_mass = 150e3 # this and propellant mass found on SpaceX web page on Starship
    propellant_mass = 249e3
    engine_mass = 18144 # mass of nerva engine (kg)
    raptor_mass = 1630 # mass of one raptor engine
    wet_mass = dry_mass + payload_mass + propellant_mass + engine_mass - raptor_mass * 6
    isp = 841
    
    propellant_1 = wet_mass * (1 - math.e**(-shipDeltaV1/(isp*g))) # propellant expended by departing burn (kg)
    propellant_2 = (wet_mass - propellant_1) * (1 - math.e**(-shipDeltaV2/(isp*g))) # propellant expended by arriving burn (kg)
    propellant_total = propellant_1 + propellant_2

    earth, times = keplerian_propagator(earthInitPos, earthInitVel, integration_time*2, integration_steps)
    mars, times = keplerian_propagator(marsInitPos, marsInitVel, integration_time*3, integration_steps)
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

    print("Transfer Time (days): "+str(integration_time/86400))
    print("Delta V at Departure (km/s): "+str(shipDeltaV1))
    print("Delta V at Arrival (km/s): "+str(shipDeltaV2))
    print("Departing Propellant Expenditure (t): "+str(propellant_1/1e3))
    print("Arriving Propellant Expenditure (t): "+str(propellant_2/1e3))
    print("Total Propellant Expenditure (t): "+str(propellant_total/1e3))
    
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
