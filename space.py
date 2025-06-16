import turtle
import math
turtle.screensize(1000,1000,"black")

counter = 0
earthRad = 150e9 # avg. distance from earth to sun (in m)
marsRad = 228e9 # avg. distance from mars to sun (in m)
scale = 1e-9 # 1 pixel = 1 million km or 1 billion m
timeScale = 86400 # 1 loop = 1 day or 86400 s

# the sun
sun = turtle.Turtle()
sun.color("yellow")
sun.shape("circle")

# the earth
earth = turtle.Turtle()
earth.color("blue")
earth.shape("circle")
earth.penup()
earth.goto(earthRad*scale,0)
earth.setheading(90)

# mars
mars = turtle.Turtle()
mars.color("red")
mars.shape("circle")
mars.penup()
mars.goto(marsRad*scale,0)
mars.setheading(90)

# the spaceship
ship = turtle.Turtle()
ship.color("white")
ship.penup()
ship.goto(earthRad*scale,0)
ship.setheading(90)

# earth text
earthText = turtle.Turtle()
earthText.hideturtle()
earthText.color("blue")
earthText.penup()

# mars text
marsText = turtle.Turtle()
marsText.hideturtle()
marsText.color("red")
marsText.penup()

# ship text
shipText = turtle.Turtle()
shipText.hideturtle()
shipText.color("white")
shipText.penup()

# planet angular velocities
earthOmega = 0.986 # 360/365.25
marsOmega = 0.524 # 360/687

# ship speed
shipVX = 0
shipVY = 11186+29800 # m/s; escape velocity + earth orbital velocity

# planet angles
earthAngle = 0
marsAngle = 0

# constants
gConstant = 6.67e-11
sunMass = 1.989e30 # kg

shipX = earthRad
shipY = 0
shipRad = earthRad

# start simulation
earth.pendown()
mars.pendown()
ship.pendown()

while True:
    
    # time notation
    if counter % 10 == 0:
        earthText.write(counter)
        marsText.write(counter)
        shipText.write(counter)
    if counter == 365:
        earthText.write(counter)
    if counter == 687:
        marsText.write(counter)
    counter += 1
    
    # earth position
    earthX = earthRad*scale*math.cos(math.radians(earthAngle))
    earthY = earthRad*scale*math.sin(math.radians(earthAngle))
    earthPos = (earthX, earthY)

    # mars position
    marsX = marsRad*scale*math.cos(math.radians(marsAngle))
    marsY = marsRad*scale*math.sin(math.radians(marsAngle))
    marsPos = (marsX, marsY)

    # ship position
    shipRad = math.sqrt(shipX**2 + shipY**2)
    shipA = -gConstant*sunMass/(shipRad**2)
    shipAX = shipA*shipX/shipRad
    shipAY = shipA*shipY/shipRad
    shipVX += shipAX*timeScale
    shipVY += shipAY*timeScale
    shipX += shipVX*timeScale
    shipY += shipVY*timeScale
    shipPos = (shipX*scale, shipY*scale)

    # update position
    earth.goto(earthPos)
    mars.goto(marsPos)
    ship.goto(shipPos)
    earthText.goto(earth.pos())
    marsText.goto(mars.pos())
    shipText.goto(ship.pos())
    
    # update angle
    earthAngle = (earthAngle + earthOmega)%360
    marsAngle = (marsAngle + marsOmega)%360
    
