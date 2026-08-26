# Mission-to-Mars

## Overview
A project simulating and evaluating propulsion systems for a future manned Mars mission using Python programs. Each file simulates a different scenario for an orbital transfer between Earth and Mars at their respective semimajor axes. The vehicle is the SpaceX Starship. Six different propulsion systems are modeled here: (1) the chemical rocket, (2) the nuclear thermal rocket, (3) the nuclear pulse rocket, (4) the solar sail, (5) the ion thruster, (6) the magnetoplasmadynamic (MPD) thruster. The programs are accompanied by a paper analyzing, comparing, and ranking these simulation results on transfer time and propellant consumption.

## Installation
Requires MatPlotLib and SciPy.
To install MatPlotLib, type into Command Prompt (for Windows) or Terminal (for Mac): ```pip install matplotlib```
To install SciPy, type: ```pip install scipy```

## Execution
To execute each file, run it in IDLE. Each simulation models the Earth, Mars and the spacecraft under gravitational acceleration of the Sun. The high-thrust simulations each apply an impulsive delta-V at the beginning of the transfer to accelerate out of Earth orbit and at the end to decelerate into Mars orbit. The low-thrust simulations do the same but also apply a low continuous delta-V in transit. Each simulation prints its results including transfer time, delta-V, and propellant consumption, and uses MatPlotLib to generate plots of the orbits. To simplify, the planets' orbits are represented as circular and coplanar, and the spacecraft is assumed to have entered orbit upon reaching the same velocity and radius from the Sun as the planet (disregarding planetary gravitational wells).

## Paper
The accompanying paper describes the propulsion systems and equations on which the simulations are based and catalogs the simulation results and orbital plots, then ranks the systems in terms of transfer time and propellant expenditure and recommends one high-thrust system that is superior to the others by both metrics. For low-thrust systems of which no one system can be identified that is superior by both metrics, the paper presents a new metric: the ratio of the difference in transfer time over the difference in propellant expenditure between a low-thrust system and its corresponding high-thrust system, referred to as the time-to-mass ratio. A higher ratio means superior performance, making it possible to identify a superior low-thrust system.
The paper can be found at this link: https://www.researchgate.net/publication/395645472_Simulation_and_Comparison_of_Propulsion_Systems_for_a_Manned_Mission_to_Mars

## Results
Simulation Plots:
<img width="470" height="508" alt="Screenshot 2026-08-25 at 10 49 00 PM" src="https://github.com/user-attachments/assets/cd563f57-5ef2-4208-9157-e17615b3cf05" />
Table of Results:
<img width="315" height="360" alt="Screenshot 2026-08-25 at 10 48 09 PM" src="https://github.com/user-attachments/assets/a114e506-2f20-450b-a565-cee6a0147d79" />

## Table of Contents
1. ```chemical_hohmann.py```: uses six Raptor chemical rockets to execute a Hohmann transfer.
2. ```chemical_fast.py```: uses six Raptor chemical rockets to expend the maximum amount of propellant in the departing burn to achieve the maximum possible transfer velocity while leaving enough propellant to enter Mars orbit.
3. ```nerva_hohmann.py```: uses one NERVA XE-Prime nuclear thermal rocket to execute a Hohmann transfer.
4. ```nerva_fast.py```: uses one NERVA XE-Prime nuclear thermal rocket to expend the maximum amount of propellant in the departing burn to achieve the maximum possible transfer velocity while leaving enough propellant to enter Mars orbit.
5. ```orion_hohmann.py```: uses one NASA 10-meter diameter Project Orion nuclear pulse engine to execute a Hohmann transfer.
6. ```orion_fast.py```: uses one NASA 10-meter diameter Project Orion nuclear pulse engine to expend the maximum amount of propellant in the departing burn to achieve the maximum possible transfer velocity while leaving enough propellant to enter Mars orbit.
7. ```solar_sail.py```: uses six Raptor chemical rockets to exit Earth orbit at Hohmann velocity, then accelerates in the direction away from the Sun using a 1-km-wide, 2.5-micrometer-thick square solar sail, then uses the Raptors again to enter Mars orbit.
8. ```ion.py```: uses one XE-Prime nuclear rocket to exit Earth orbit at Hohmann velocity, then accelerates in the direction of its current velocity using 40 ion thrusters, then uses the XE-Prime again to enter Mars orbit. (The NTR was chosen to provide electrical power for the ion thrusters.)
9. ```plasma.py```: uses one XE-Prime nuclear rocket to exit Earth orbit at Hohmann velocity, then accelerates in the direction of its current velocity using 40 of a 1989 NASA prototype of an MPD thruster, then uses the XE-Prime again to enter Mars orbit.
10. ```space.py```: a prototype simulation using Turtles to simulate Earth, Mars, and the spacecraft. Not part of the main project.
