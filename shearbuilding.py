import numpy as np
import Fourier 
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation



subSoil =  [
           {
            "A" :[1.00  ,0.05   ,0.25   ,1.2],
            "B" :[1.35  ,0.05   ,0.25   ,1.2],
            "C" :[1.50  ,0.10   ,0.25   ,1.2],
            "D" :[1.80  ,0.10   ,0.30   ,1.2],
            "E" :[1.60  ,0.05   ,0.25   ,1.2]
            },
            {
            "A" :[1.00  ,0.15   ,0.4    ,2],
            "B" :[1.20  ,0.15   ,0.5    ,2],
            "C" :[1.15  ,0.20   ,0.6    ,2],
            "D" :[1.35  ,0.20   ,0.8    ,2],
            "E" :[1.40  ,0.15   ,0.5    ,2]
            }
          ]

class ShearBuilding:
    def __init__(self, I=1, R=5,SoilType="C",SpectrumType=0,ag=0.15) -> None:
        #structure
        self.m = []
        self.k = []
        self.M_mat =None
        self.K_mat = None

        #eigen values 
        self.eValues = None     
        self.eVecs = None   # matrix of eigen  vector
        self.PF = None
        self.D = None
        self.w = None
        self.Tn = None

        #Building Nature
        self.I = I      # importance factor
        self.R = R      # over strength
        self.SoilType = SoilType
        self.SpectrumType = SpectrumType
        self.ag = ag
        
        #modal analysis
        self.EF = None
        self.modalMass = None
        self.BaseShear = 0
        
        # time history
        self.A = None       # factors of dis for natural frequncy 
        self.B=None         # factors of v for natural frequncy 
        
        self.MT = None      #Eigen space Tramsformtion
        self.MTI = None      #Eigen space Tramsformtion
        self.PR_COS = None      #Particular responce spectrum
        self.PR_SIN = None      #Particular responce spectrum
       
        self.beta = None
        self.Pag =None
        self.f = 0.01

        self.fig, self.ax = plt.subplots()


    def add_floor(self, mass , stiffness):
        self.m.append(mass)
        self.k.append(stiffness)

    def build(self):
        n = len(self.m)
        self.K_mat = np.zeros((n,n))
        self.M_mat = np.zeros((n,n))

        for i in range(0, n-1):
            self.K_mat[i,i] = self.k[i]+self.k[i+1]
            self.K_mat[i,i+1] = -self.k[i+1]
            self.K_mat[i+1,i] = -self.k[i+1]
            self.M_mat[i,i] = self.m[i]

        self.K_mat[n-1,n-1] = self.k[n-1]
        self.M_mat[n-1,n-1] = self.m[n-1]
        

        mat = np.matmul( np.linalg.inv(self.M_mat), self.K_mat)
        self.eValues, self.eVecs = np.linalg.eig(mat)

        for i in range(0, len(self.m)):
            m1  = np.matmul( np.transpose( self.eVecs[i]) , self.M_mat)
            a  = np.matmul( m1,self.eVecs[i])
            self.eVecs[i] /= a**0.5

        self.w = np.sqrt(self.eValues)
        self.Tn = 2 * np.pi / self.w

        self.MT = (self.eVecs)
        self.MTI = np.linalg.inv(self.MT)
        self.calcLateralLoad()


# Modal  analysis:

    def  ResponceSpectrum(self, T):
        r = 0
        S = subSoil[self.SpectrumType][self.SoilType][0]
        TB = subSoil[self.SpectrumType][self.SoilType][1]
        TC = subSoil[self.SpectrumType][self.SoilType][2]
        TD = subSoil[self.SpectrumType][self.SoilType][3]

        if(T < TB):
            r = self.ag * self.I *S * (2/3 + T/TB * (2.5/self.R - 2/3))

        elif(T < TC):
            r = self.ag * self.I *S * (2.5/self.R )

        elif(T < TD):
            r = self.ag * self.I *S * (2.5/self.R ) * TC/T

        else:
            r = self.ag * self.I *S * (2.5/self.R ) * TC * TD/T**2
        return r

    def ParticipationFactor(self):
        self.PF = np.zeros(len(self.m))
        
        for i in range(0, len(self.m)):
            self.PF[i] = 0

            for j in range(0, len(self.m)):
                self.PF[i] += self.eVecs[i][j] * self.m[j]

        self.PF = self.PF 
        
        return self.PF
    

    def calcLateralLoad(self):
        n =len(self.m)
        Ri = np.zeros(n)
        self.modalMass= np.zeros(n)
    
        Pr = self.ParticipationFactor()
        self.EF = np.zeros((n,n))

        self.BaseShear = 0

        for i in range(0, len(self.m)):
            si = self.ResponceSpectrum(self.Tn[i])
            Ri  = self.eVecs[i] * Pr[i] * si*self.m
           
            self.EF[i] = Ri
            self.modalMass[i] = abs(np.sum(Ri)) / si
            BaseShear = (np.sum(Ri))
            self.BaseShear += BaseShear**2
        
        self.BaseShear = self.BaseShear**0.5

        return self.BaseShear

# Time History Analysis Data :

    def _initialConditions(self, y0, u0):
        A = np.matmul( self.MTI,y0)
        B = -np.matmul( self.MTI,u0/self.w)
        n = len(self.m)
        
        self.B = B
        self.A = A

    def _calcHomogenous_y(self, t):
        return np.matmul( self.MT , self.A * np.cos(self.w * t)) + np.matmul( self.MT , self.B * np.sin(self.w * t)) 
        
    def _calcHomogenous_v(self, t):
        return np.matmul( self.MT , -self.A * self.w*np.sin(self.w * t)) + np.matmul( self.MT , self.B *self.w* np.cos(self.w * t)) 
        
    
    def _solveParticularResponce(self,i, beta, pr):
        """
            beta ground speed angular frequncy R = Ug * cos(beta * t)
        """
        n = len(self.m) 
        I = np.identity(n)
        
        R = np.linalg.inv(self.D-beta*beta * I)
        R = np.matmul(R,self.MTI)
        R = np.matmul(self.MT,R)
        
        R = np.matmul(R,pr)
        return R
    
    def _ParticularResponce(self,t):
        a = 0
        n = len(self.m) 

        for i in range(0,len(self.beta)):
            a +=  self.PR_COS[i] * np.cos(self.beta[i] * t)
            a +=  self.PR_SIN[i] * np.sin(self.beta[i] * t)

        return a

    def solve(self , betas, PrCOS,PrSIN , u0,v0):
        n = len(self.m) 
        self.D = np.zeros((n,n))
        self.PR_COS = np.zeros((len(betas),n))
        self.PR_SIN = np.zeros((len(betas),n))

        a = np.ones(n)
        np.fill_diagonal(self.D,self.eValues)
        
        self.beta =np.array( betas)

        for i in range(0,len(betas)):
            self.PR_COS[i] = self._solveParticularResponce(i,betas[i],PrCOS[i]*a)       
            self.PR_SIN[i] = self._solveParticularResponce(i,betas[i],PrSIN[i]*a)       
        
        self._initialConditions(u0 -self._ParticularResponce( 0),v0 )
        return
      
    def calcResponce(self,t): 
        return self._ParticularResponce( t)+ self._calcHomogenous_y(t)

    
# data output:
    
    def __str__(self):
       
        a1 = f"""\n\t*********************************\n
SHEAR BUILDING STRUCTURE ANALYSIS
SHEAR BUILDING FLOORs Masss
{self.m}
\n SHEAR BUILDING FLOORs STIFFNESS
{self.k}
"""
       
        a1 += f"""
***************STIFFNESS******************
{self.K_mat}
\n***************MASS******************
{self.M_mat}
eigen vectors are:........
{self.eVecs}

Transformation matrix inverse are:........
{self.MT}

"""
        a2 = f"""
\nPeriodic times are:........ 
{self.Tn}
\nParticipation Factor are:........ 
{self.PF}\n

modal masses Factor are:........ 
{np.round(self.modalMass,2)}\n

 Mode importance Ratio are:........ 
{np.round( (self.modalMass) / (np.sum(self.m))*100,2)}
IMPORTANCE FACTOR = {self.I} 
OVERSTRENGTH FACTOR = {self.R} 
SOIL ACCELERATION = {self.ag} 
Lateral Loads = 
{np.round( self.EF,2)} 

BASE SHEAR =
{round(self.BaseShear,2)} 
  
        \tEnd \n"""

        return a1+a2
        
    def animate_displacement(self, dis=10.0,step=0.1, interval=50):
        # Initialize plot
        self.ax.set_xlim(-dis, dis)
        self.ax.set_ylim(0, len(self.m) + 1)
        self.ax.set_xlabel('displacements')
        self.ax.set_ylabel('Floor')
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], marker='o', linestyle='-')
        self.ax.scatter([0], [0], color='red')  # Adjust color and size as needed

        # Animation function
        def update(frame):
            t = frame * step  # Example time increment (adjust as needed)
            displacements = self.calcResponce(t)
            displacements =np.insert( displacements,0,0)

            y_values = np.arange(0, len(displacements))

            self.line.set_data(displacements, y_values)
            print(f"Frame: {frame}, Time: {round(t,2)}", end='\r') # Print frame number and time
              
            return self.line,

        # Create animation
        anim = FuncAnimation(self.fig, update, frames=int(interval / step), interval=interval, blit=True)
        plt.show()

        


if __name__ =='__main__':
    
    data = np.array( Fourier.read_csv_file("SANTA MONICA.csv")).transpose()
    x = data[0]
    y = data[1]
    A,B,w = Fourier.fourier_coefficients(x,y,1000)

    plt.figure()
    plt.plot(x, y, label="Original Function")
    plt.plot(x, Fourier.fourier_series(x,A,B), label="Fourier Series Approximation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Fourier Series Approximation")
    plt.legend()
    plt.show()

           
    building =  ShearBuilding(1,3,"C",0,0.15*9.81)
     
    building.add_floor(200,18*10**6)
    building.add_floor(200,18*10**6)
    building.add_floor(200,18*10**6)
    building.add_floor(200,18*10**6)
    building.add_floor(200,18*10**6)
    building.add_floor(200,18*10**6)
    building.add_floor(200,18*10**6)
    building.add_floor(200,1.8*10**6)
    building.add_floor(200,1.8*10**6)
    building.add_floor(200,1.8*10**6)
    building.add_floor(200,1.8*10**6)
    building.add_floor(200,1.8*10**6)
    building.add_floor(200,1.8*10**6)
    building.add_floor(200,1.8*10**6)
    building.add_floor(200,1.8*10**6)
    building.add_floor(200,1.8*10**6)
    
    
    building.build()
    building.solve(w,A,B,np.zeros(len(building.m)),np.zeros(len(building.m)))
    
    building.animate_displacement(0.05,0.02,60)
    
