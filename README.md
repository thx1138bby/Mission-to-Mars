# Mission-to-Mars

## Overview
Simulations for the paper "A Simulation and Comparison of Propulsion Systems for a Manned Mission to Mars" evaluating different space propulsion systems for a future manned Mars mission. The mission is one-way from Earth to Mars at their respective semimajor axes. The vehicle is the SpaceX Starship. The propulsion systems include (1) the chemical rocket, (2) the nuclear thermal rocket, (3) the nuclear pulse engine, (4) the solar sail, (5) the ion thruster, and (6) the magnetoplasmadynamic (MPD) thruster.

## Abstract
Recent developments suggest a manned mission to Mars as a realistic near future possibility, therefore requiring additional consideration into which propulsion system to use. Examples include the traditional chemical rocket; the nuclear pulse engine, which propels a spacecraft using nuclear explosions; the ion thruster, which accelerates propellant using electrically charged grids; and others. This paper evaluates six such propulsion systems using Python computer simulations. Each simulation models a SpaceX Starship spacecraft modified to use one of the six propulsion systems, calculating and returning values for transfer time and propellant mass consumption. These results are analyzed and ranked to determine how the different systems would compare in performance in an actual mission, specifically in the aforementioned criteria of travel time and propellant expenditure. 

## Dependencies and Installation
Requires MatPlotLib and SciPy.
To install MatPlotLib type into Terminal: ```pip install matplotlib```
To install SciPypip type into Terminal: ```pip install scipy```

## Execution
Each of these files simulates a different transfer scenario from Earth to Mars. To execute, run each file in IDLE. For the High-thrust engines, an impulsive delta-V is applied at the beginning and end of the transfer arcs. During the transfer, the spacecraft is assumed to be under the 2-body gravitational acceleration of the Sun. For the simulations utilizing low-thrust, an impulsive delta-V is applied again, but during the transfer arcs keplerian motion is applied along with the low thrust acceleration which is typically velocity aligned. 

## Table of Contents
1. ```chemical_fast.py```: using 6 Raptor chemical rockets, the spacecraft expends the maximum possible amount of propellant at the departing burn to produce the maximum transfer velocity while still leaving enough propellant to enter Mars orbit.
2. ```chemical_hohmann.py```: using 6 Raptor chemical rockets, the spacecraft executes a Hohmann transfer from Earth to Mars.
3. ```ion.py```: the spacecraft exits Earth orbit at Hohmann velocity using the XE-Prime nuclear thermal rocket and accelerates in the same direction as its velocity using 40 ion thrusters.
4. ```nerva_fast.py```: using a NERVA XE-Prime nuclear thermal rocket, the spacecraft expends the maximum possible amount of propellant at the departing burn to produce the maximum transfer velocity while still leaving enough propellant to enter Mars orbit.
5. ```nerva_hohmann.py```: using a NERVA XE-Prime nuclear thermal rocket, the spacecraft executes a Hohmann transfer from Earth to Mars.
6. ```orion_fast.py```: using a NASA 10-m diameter Project Orion nuclear pulse engine, the spacecraft expends the maximum possible amount of propellant at the departing burn to produce the maximum transfer velocity while still leaving enough propellant to enter Mars orbit.
7. ```orion_hohmann.py```: using a NASA 10-m diameter Project Orion nuclear pulse engine, the spacecraft executes a Hohmann transfer from Earth to Mars.
8. ```plasma.py```: the spacecraft exits Earth orbit at Hohmann velocity using the XE-Prime nuclear thermal rocket and accelerates in the same direction as its velocity using 40 of a 1989 NASA prototype of a MPD thruster.
9. ```solarsail.py```: the spacecraft exits Earth orbit at Hohmann velocity using 6 Raptor chemical rockets and accelerates in the direction away from the Sun using a 1-km-wide, 2.5-micrometer-thick square solar sail.
10. ```space.py```: a prototype simulation using Turtles to simulate Earth, Mars, and the spacecraft. Not cited in the paper.
