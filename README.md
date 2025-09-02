# Mission-to-Mars
Simulations for the paper "A Simulation and Comparison of Propulsion 
Systems for a Manned Mission to Mars" evaluating different space propulsion systems for a future manned Mars mission.
Requires MatPlotLib and Scipy. To install type in terminal: pip install matplotlib, pip install scipy


## Abstract
Recent developments suggest a manned mission to Mars as a realistic near future 
possibility, therefore requiring additional consideration into which propulsion system to use. 
Examples include the traditional chemical rocket; the nuclear pulse engine, which propels a 
spacecraft using nuclear explosions; the ion thruster, which accelerates propellant using 
electrically charged grids; and others. This paper evaluates six such propulsion systems using 
Python computer simulations. Each simulation models a SpaceX Starship spacecraft modified to 
use one of the six propulsion systems, calculating and returning values for transfer time and 
propellant mass consumption. These results are analyzed and ranked to determine how the 
different systems would compare in performance in an actual mission, specifically in the 
aforementioned criteria of travel time and propellant expenditure. 

## Execution
Each of these files simulates a different transfer scenario from Earth to Mars. To execute, run each of the files. For the High-thrust engines, an impulsive delta-V is applied at the beginning and end of the transfer arcs. During the transfer, the spacecraft is assumed to be under the 2-body gravitational acceleration of the Sun. For the simulations utilizing low-thrust, an impulsive delta-V is applied again, but during the transfer arcs keplerian motion is applied along with the low thrust acceleration which is typically velocity aligned. 
