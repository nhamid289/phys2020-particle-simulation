import numpy as np
from matplotlib import pyplot as plt

def main():
    # Simulation parameters
    T_i= 300; #Temperature in Kelvin
    k=1.38e-23; #Boltzmann constant
    m= 1e-26;    #Mass of a particle in kg
    L= 0.0001;    #length of side in meters
    N = 100 #number of particles
    dis = np.sqrt((k * T_i )/m) 

    #Particle position - randomly locating particle
    pos = L * (np.random.random_sample([N,2]))

    #Particle velocity - select initial particle velocity components
    vel =  dis * (np.random.standard_normal([N,2]))
  
    Tau = L / np.amin(np.abs(vel)) # Time for particle to traverse box. 
    tmax = Tau * 1 #simulation time i.e how many times the particles collide

    dt = Tau / 10000 # time step i.e.
    N_time = round(tmax/dt) # number of time steps in the simulation used to define length of loop
    
    fig = plt.figure(figsize=(8,8), dpi = 80)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    J = np.zeros(N_time)
    for t in np.arange(N_time):
        # move
        pos +=  vel * dt
        J[t] = 2 * m * np.sum(vel[pos >= L]) # Searching for any particles which have hit the right side of the box, then calculating and storing the total impulse from them
        vel = np.where(pos >= L, -vel, vel) #checks if particle is outside the box on the right,if so reversing the direction    
        vel = np.where(pos <= 0, -vel, vel) # checks if the particle is outside the box on the left, if so reversing the direction left


        # Plotting of the particles, done every 5 time steps to save computational power. 
        if ( t % 5 == 0):
            plt.cla()
            plt.plot(pos[:,0],pos[:,1],'bo')
            ax.set(xlim=(0, L), ylim=(0, L))
            ax.set_aspect('equal')	
            plt.pause(0.001)
        
    plt.show()

    Jtc = np.cumsum(np.abs(J), axis = 0)
    Fbar = Jtc[-1] / tmax
    P = Fbar / L

    plt.plot(np.linspace(0,tmax,N_time),Jtc)
    print("The total impulse imparted on the wall is " + str(Jtc[-1]) + " Newton meters. ")
    print("The average on the force on the wall is " + str(Fbar) + " Newtons. ")
    print("The pressure in the box is " + str(P) + " . ")
    plt.show()


main()