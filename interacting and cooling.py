import numpy as np
from matplotlib import pyplot as plt

def main():
    # Basic Simulation parameters
    T_i = 500; # initial Temperature in Kelvin
    k= 1.38e-23; #Boltzmann constant
    m= 6.63e-26;    #Mass of a particle in kg
    N = 64 #number of particles
    mi = 1/m

    #Extra Parameters
    eps = 1.95e-21
    sig = 0.34e-9
    A = 4 * eps * sig**12
    B = 4 * eps * sig**6
    L= 15 * sig  #length of side in meters
    lmbda = 0.9995
    dis = np.sqrt((k * T_i )/m) 

    def r(xdis,ydis):
        return np.sqrt(xdis**2 + ydis**2)
    def r2(xdis,ydis):
        return xdis**2 + ydis**2
    def Fx(xdis,ydis):
        #return (-24*eps)*((2*xdis*sig**12)*(xdis**2 + ydis**2)**(-7) - (xdis*sig**6)*((xdis)**2+(ydis)**2)**(-4))
        return (-12 * A * xdis ) * r2(xdis,ydis) ** (-7) + (6* B *xdis ) * r2(xdis,ydis) ** (-4)
    def Fy(xdis,ydis):
        #return (-24*eps)*((2*ydis*sig**12)*(xdis**2 + ydis**2)**(-7) - (ydis*sig**6)*((xdis)**2+(ydis)**2)**(-4))
        return (-12 * A * ydis) * r2(xdis,ydis) ** (-7)  + (6* B * ydis) * r2(xdis,ydis) ** (-4)


    Tau = L / dis # Time for particle to traverse box. 
    tmax = Tau * 10 #simulation time i.e how many times the particles collide
    dt = 0.01 * sig * np.sqrt(m/eps) # time step i.e.
    #dt = Tau / 3000
    N_time = round(tmax/dt) # number of time steps in the simulation used to define length of loop
    TC = m/(2* N * k)
    trials = 1
    #Temperature parameters
    T = np.zeros([trials, int(np.ceil(tmax/dt))])
    a = np.arange(0,tmax,dt)
    Pres = np.zeros([trials, N_time])
    # Time parameters

    for tn in np.arange(trials):

        #Placing particles in a grid
        gridedge = np.linspace(0.01*L,0.99*L, int(np.sqrt(N)))
        icord, jcord = np.meshgrid(gridedge, gridedge)
        pos = np.column_stack((icord.reshape(-1),jcord.reshape(-1)))


        #Particle velocity - select initial particle velocity components
        vel = dis * (np.random.standard_normal([N,2]))
        acc = np.zeros([N,2])
        J = np.zeros(N_time)

        #Setting up the figure
        fig = plt.figure(figsize=(8,8), dpi = 80)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for t in np.arange(N_time):
            vel = vel * lmbda  
            pos += vel * dt + 0.5 * acc * dt**2
            vel += 0.5 * acc * dt 

            acc.fill(0)

            for i in np.arange(N):


                for j in np.arange(i-1):

                    x = pos[i,0] - pos[j,0]
                    y = pos[i,1] - pos[j,1]

                    
                    acc[i,0] += mi * -Fx(x,y)
                    acc[i,1] += mi * -Fy(x,y)
                    acc[j,0] += mi * Fx(x,y)
                    acc[j,1] += mi * Fy(x,y)
                
            # updating the velocities based on calculated accelerations
            vel += 0.5 * acc * dt 

            vel = np.where(pos >= L, -vel, vel) #checks if particle is outside the box on the right,if so reversing the direction    
            vel = np.where(pos <= 0, -vel, vel) # checks if the particle is outside the box on the left, if so reversing the direction left

            J[t] = 2 * m * np.sum(vel[pos >= L]) # Searching for any particles which have hit the right side of the box, then calculating and storing the total impulse from them

            #calculating temperature
            T[tn,t] =  TC * np.sum(np.sum(vel ** 2, axis = 0))
            # Plotting of the particles, done every 5 time steps to save computational power. 
            if ( t % 50 == 0):
                plt.cla()
                plt.plot(pos[:,0],pos[:,1],'bo', markersize = 22)
                ax.set(xlim=(0, L), ylim=(0, L))
                ax.set_aspect('equal')	
                plt.pause(0.0000000000000001)
            
        plt.show()

        Jtc = np.cumsum(np.abs(J), axis = 0)

        for t in np.arange(N_time):
            Pres[tn,t] = Jtc[t] / (2* t * dt * L)

        
        #print("The total impulse imparted on the wall is " + str(Jtc[-1]) + " Newton meters. ")
         
    for tn in np.arange(trials): 
        plt.plot(a.reshape(-1,1),T[tn,:], label = "Trial" + str(tn+1))
    
    plt.plot(a.reshape(-1,1),np.mean(T, axis = 0), label = "Average")
    plt.axis([0, tmax, 0, 2 * int(round(1.1*np.max(T)))])
    plt.legend()
    plt.xlabel("Time(s)")
    plt.ylabel("Temperature (K)")
    plt.show()

    for tn in np.arange(trials):
        plt.plot(np.linspace(0,tmax,N_time),Pres[tn,:], label = "Trial" + str(tn+1))
    plt.plot(np.linspace(0,tmax,N_time), np.mean(Pres, axis = 0), label = "Average")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.title("Pressure over time")
    plt.show()
    
main()