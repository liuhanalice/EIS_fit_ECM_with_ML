import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import constants
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
import scipy.io


###### Define Essential Elements ######
# Angular frequency: omega = 2 * pi * f
# Impedance:
#  - Resistor: Z_R = R
#  - Constant Phase Element (CPE): Z_Q = 1 / (Q * (1j * omega) ** alpha
#  - Infinite Watburg Element: Z_W = sigma * sqrt(2) / sqrt(j * omega)
def F_range(initial_frequency,last_frequency,number_of_point=100): 
    """ 
        Define frequency range in log scale with the output of frequency in [Hz] and angular frequency in [s^-1]
        
        :param initial_frequency: initial frequency [Hz]
        :param last_frequency: last frequency [Hz]
        :param number_of_point: number of point (default is 100 points)
        :return array: array[0] = log scale of angular frequency [s^-1], array[1] = log scale of Frequency [Hz]

    """
    frequency_Hz = np.logspace(np.log10(initial_frequency), np.log10(last_frequency),number_of_point, endpoint=True)
    angular_frequency = 2 * np.pi * frequency_Hz
    return angular_frequency, frequency_Hz


def Z_R(resistance):
    """
        Define impedance of resistor (R)
        
        :param resistance: R [Ohm]
        :return resistor_impedance: Impedance of resistor [Ohm]
    """
    resistor_impedance = resistance
    return resistor_impedance


def Z_Q(non_ideal_capacitance,ideality_factor,angular_frequency):
    """
        Define impedance of constant phase element (CPE)
        
        :param non_ideal_capacitance: Q [s^alpha Ohm^-1]
        :param ideality_factor: alpha [n/a.]
        :param angular_frequency: omega [s^-1]
        :return CPE_impedace: Impedance of the CPE [Ohm]
    """
    CPE_impedace = 1/(non_ideal_capacitance * (angular_frequency*1j)**ideality_factor)
    return CPE_impedace


def Z_W(sigma,angular_frequency):
    """
        Define impedance of infinite Watburg element (W)
        
        :param sigma: Warburg coefficient [Ohm s^-1/2]
        :param angular_frequency: omega [s^-1]
        :return W_impedace: Impedance of the Warburg Element [Ohm]
    """
    W_impedace =( sigma * np.sqrt(2) ) / np.sqrt(1j*angular_frequency)  
    return W_impedace

def Z_L(inductance, angular_frequency):
    """
        Define impedance of inductor (L)
        
        :param inductance: L [H]
        :return resistor_impedance: Impedance of inductor [Ohm]
    """
    L_impedance = 1j*angular_frequency*inductance
    return L_impedance

###### Define Essential Functions ######
def log_rand(initial_gen,last_gen,size_number):
    """ 
        Generate random number in log scale

        :param initial_gen: initial value
        :param last_gen: last value
        :param size_number: number of random number to be generated
        :return log_array: array of random number generated in log scale
    """
    initial_v = np.log(initial_gen)
    last_v = np.log(last_gen)
    log_array = np.exp( ( initial_v + ( last_v - initial_v ) * np.random.rand( size_number ) ) )
    return log_array

def lin_rand(initial_gen,last_gen,size_number):
    """ 
        Generate random number in linear scale

        :param initial_gen: initial value
        :param last_gen: last value
        :param size_number: number of random number to be generated
        :return lin_array: array of random number generated in linear scale
    """
    lin_array = initial_gen + ( last_gen - initial_gen ) * np.random.rand( size_number )
    return lin_array

def nor_rand(mu, size_number, sigma = None):
    """ 
    Generate random numbers from a normal distribution

    :param mu: mean of the normal distribution
    :param sigma: standard deviation of the normal distribution
    :param size_number: number of random numbers to generate
    :return norm_array: array of normally distributed random numbers
    """
    if sigma is None:
        sigma = 0.2*mu
    norm_array = np.random.normal(loc=mu, scale=sigma, size=size_number)
    return norm_array

###### Array Generator Functions ######
def genZR(size_number,number_of_point,resistance):
    """Generate array of impedance of resistors"""
    ZR= np.zeros((size_number,number_of_point), dtype=complex)
    for idx in range(size_number):
        for idx2 in range(number_of_point): 
            ZR[idx][idx2] = Z_R(resistance[idx])
    return ZR

def genZQ(size_number,number_of_point,non_ideal_capacitance,ideality_factor,angular_frequency):
    """Generate array of impedance of CPEs"""
    ZQ= np.zeros((size_number,number_of_point), dtype=complex)
    for idx in range(size_number):
        for idx2 in range(number_of_point): 
            ZQ[idx][idx2] = Z_Q(non_ideal_capacitance[idx],ideality_factor[idx],angular_frequency[idx2])
    return ZQ

def genZW(size_number,number_of_point,sigma,angular_frequency):
    """Generate array of impedance of Warburgs"""
    ZW= np.zeros((size_number,number_of_point), dtype=complex)
    for idx in range(size_number):
        for idx2 in range(number_of_point): 
            ZW[idx][idx2] = Z_W(sigma[idx],angular_frequency[idx2])
    return ZW

def genZL(size_number,number_of_point,inductance):
    """Generate array of impedance of resistors"""
    ZL= np.zeros((size_number,number_of_point), dtype=complex)
    for idx in range(size_number):
        for idx2 in range(number_of_point): 
            ZL[idx][idx2] = Z_L(inductance[idx], angular_frequency[idx2])
    return ZL

###### Visulaization Functions ######
plt.rcParams['axes.formatter.min_exponent'] = 4

class myformatter(ticker.LogFormatter):
    
    def _num_to_string(self, x, vmin, vmax):
        if x > 10000:
            s = '%1.0e' % x
        elif x < 1 and x >= 0.001:
            s = '%g' % x
        elif x < 0.001:
            s = '%1.0e' % x
        else:
            s = self._pprint_val(x, vmax - vmin)
        return s
    
xfmt = myformatter(labelOnlyBase=False, minor_thresholds=(4, 0.5))
yfmt = myformatter(labelOnlyBase=False, minor_thresholds=(3, 0.5))


class Zplot:
    """ Plotting EIS Spectrum"""
    def full(ZZ,frequency,param,nrow=1,examp=1):
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig , axs =plt.subplots(figsize=(700*px, 200*px*nrow),nrows=nrow,ncols=3) 
        
        if nrow==1:
            phase = np.degrees(np.arctan(ZZ[examp-1].imag/ZZ[examp-1].real))
            mag = np.absolute(ZZ[examp-1])
            paramtxt=""
            if param != "" :
              for idx in range(len(param[examp-1])):
                  if idx==4:
                      paramtxt=paramtxt+" \n "                    
                  paramtxt=paramtxt+(format(param[examp-1][idx],".3g"))+" | "           
            
            axs[0].plot(ZZ[examp-1].real,-ZZ[examp-1].imag)
            axs[0].set_title(paramtxt)
            axs[0].set_xlabel("Z' (\u03A9)")
            axs[0].set_ylabel("-Z'' (\u03A9)")

            axs[1].plot(frequency,phase)
            axs[1].set_xscale('log')
            axs[1].set_title("Frequency vs. Phase")
            axs[1].set_xlabel("Frequency (Hz)")
            axs[1].set_ylabel("Phase (\u03B8)")
            axs[1].xaxis.set_minor_formatter(xfmt)

            axs[2].plot(frequency,mag)
            axs[2].set_xscale('log')
            axs[2].xaxis.set_minor_formatter(xfmt)
            axs[2].set_title("Frequency vs. |Z|")
            axs[2].set_xlabel("Frequency (Hz)")
            axs[2].set_ylabel("|Z| (\u03A9)")

            plt.tight_layout()

        elif nrow>1:

            for row in range(nrow):
                fq = frequency
                phase = np.degrees(np.arctan(ZZ[row+examp-1].imag/ZZ[row+examp-1].real))
                mag = np.absolute(ZZ[row+examp-1])
                paramtxt=""
                if param != "" :
                  for idx in range(len(param[row+examp-1])):
                      if idx==4:
                          paramtxt=paramtxt+" \n "   
                      paramtxt=paramtxt+(format(param[row+examp-1][idx],".3g"))+" | "  
                
                axs[row,0].text(0.1, 0.9, row+examp, horizontalalignment='center', verticalalignment='center', transform=axs[row,0].transAxes)

                axs[row,0].plot(ZZ[row+examp-1].real,-ZZ[row+examp-1].imag)
                axs[row,0].set_title(paramtxt)
                axs[row,0].set_xlabel("Z' (\u03A9)")
                axs[row,0].set_ylabel("-Z'' (\u03A9)")

                axs[row,1].plot(frequency,phase)
                axs[row,1].set_xscale('log')
                axs[row,1].set_title("Frequency vs. Phase")
                axs[row,1].set_xlabel("Frequency (Hz)")
                axs[row,1].set_ylabel("Phase (\u03B8)")
                axs[row,1].xaxis.set_minor_formatter(xfmt)

                axs[row,2].plot(frequency,mag)
                axs[row,2].set_xscale('log')
                axs[row,2].xaxis.set_minor_formatter(xfmt)
                axs[row,2].set_title("Frequency vs. |Z|")
                axs[row,2].set_xlabel("Frequency (Hz)")
                axs[row,2].set_ylabel("|Z| (\u03A9)")

                plt.tight_layout()
            return

    def point(ZZ,frequency,nrow=1,examp=1):
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig , axs =plt.subplots(figsize=(700*px, 200*px*nrow),nrows=nrow,ncols=4) 
        if nrow==1:
            phase = np.degrees(np.arctan(ZZ[examp-1].imag/ZZ[examp-1].real))
            mag = np.absolute(ZZ[examp-1])                

            axs[0].plot(ZZ[examp-1].real)
            axs[0].set_title("Z'")
            axs[0].set_xlabel("Point")
            axs[0].set_ylabel("Z' (\u03A9)")

            axs[1].plot(-ZZ[examp-1].imag)
            axs[1].set_title("Z''")
            axs[1].set_xlabel("Point")
            axs[1].set_ylabel("-Z'' (\u03A9)")

            axs[2].plot(phase)
            axs[2].set_title("Phase")
            axs[2].set_xlabel("Point")
            axs[2].set_ylabel("Phase (\u03B8)")
            axs[2].xaxis.set_minor_formatter(xfmt)

            axs[3].plot(mag)
            axs[3].xaxis.set_minor_formatter(xfmt)
            axs[3].set_title("|Z|")
            axs[3].set_xlabel("Point")
            axs[3].set_ylabel("|Z| (\u03A9)")

            plt.tight_layout()

        elif nrow>1:
            for row in range(nrow):
                fq = frequency
                phase = np.degrees(np.arctan(ZZ[row+examp-1].imag/ZZ[row+examp-1].real))
                mag = np.absolute(ZZ[row+examp-1])               
                axs[row,0].text(0.1, 0.9, row+examp, horizontalalignment='center', verticalalignment='center', transform=axs[row,0].transAxes)

                axs[row,0].plot(ZZ[row+examp-1].real)
                axs[row,0].set_title("Z'")
                axs[row,0].set_xlabel("Point")
                axs[row,0].set_ylabel("Z' (\u03A9)")

                axs[row,1].plot(-ZZ[row+examp-1].imag)
                axs[row,1].set_title("Z''")
                axs[row,1].set_xlabel("Point")
                axs[row,1].set_ylabel("-Z'' (\u03A9)")

                axs[row,2].plot(phase)
                axs[row,2].set_title("Phase")
                axs[row,2].set_xlabel("Point")
                axs[row,2].set_ylabel("Phase (\u03B8)")
                axs[row,2].xaxis.set_minor_formatter(xfmt)

                axs[row,3].plot(mag)
                axs[row,3].xaxis.set_minor_formatter(xfmt)
                axs[row,3].set_title("|Z|")
                axs[row,3].set_xlabel("Point")
                axs[row,3].set_ylabel("|Z| (\u03A9)")

                plt.tight_layout()

        return


##### Circuit Simulation #####
def sim_cir1():
    """ Simulate circuit 1: R1 + (R2 || Q1)"""
    R1=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr1= genZR(size_number,number_of_point,R1)

    R2=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr2= genZR(size_number,number_of_point,R2)


    ideality_factor1=np.round(lin_rand(alpha_range[0],alpha_range[1],size_number),3)
    Q1=log_rand(q_range[0],q_range[1],size_number)
    Zq1= genZQ(size_number,number_of_point,Q1,ideality_factor1,angular_frequency)

    Zsum=Zr1 + 1 / ( 1 / Zr2 + 1 / Zq1 ) 
    Zparam=[]
    for idx in range(size_number):
        Zparam.append([R1[idx],R2[idx],ideality_factor1[idx],Q1[idx]])
    
    return Zsum,np.array(Zparam)


def sim_cir2():
    """ Simulate circuit 2: R1 + (R2 || Q1) + (R3 || Q2)"""
    R1=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr1= genZR(size_number,number_of_point,R1)

    R2=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr2= genZR(size_number,number_of_point,R2)

    ideality_factor1=np.round(lin_rand(alpha_range[0],alpha_range[1],size_number),3)
    Q1=log_rand(q_range[0],q_range[1],size_number)
    Zq1= genZQ(size_number,number_of_point,Q1,ideality_factor1,angular_frequency)

    R3=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr3= genZR(size_number,number_of_point,R3)

    ideality_factor2=np.round(lin_rand(alpha_range[0],alpha_range[1],size_number),3)
    Q2=log_rand(q_range[0],q_range[1],size_number)
    Zq2= genZQ(size_number,number_of_point,Q2,ideality_factor1,angular_frequency)
    
    Zsum=Zr1 + 1 / ( 1 / Zr2 + 1 / Zq1 ) + 1 / ( 1 / Zr3 + 1 / Zq2 ) 
    
    Zparam=[]
    for idx in range(size_number):
        Zparam.append([R1[idx],R2[idx],R3[idx],ideality_factor1[idx],Q1[idx],ideality_factor2[idx],Q2[idx]])    

    return Zsum,np.array(Zparam)

def sim_cir3():
    """ Simulate circuit 3: R1 + ( (R2 + Z ) || Q1)"""
    R1=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr1= genZR(size_number,number_of_point,R1)

    ideality_factor1=np.round(lin_rand(alpha_range[0],alpha_range[1],size_number),3)
    Q1=log_rand(q_range[0],q_range[1],size_number)
    Zq1= genZQ(size_number,number_of_point,Q1,ideality_factor1,angular_frequency)

    R2=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr2= genZR(size_number,number_of_point,R2)

    sigma=log_rand(sigma_range[0],sigma_range[1],size_number)
    Zw=genZW(size_number,number_of_point,sigma,angular_frequency)
    
    Zsum=Zr1 + 1 / ( 1 / Zq1 + 1 / ( Zr2 + Zw ) )

    Zparam=[]
    for idx in range(size_number):
        Zparam.append([R1[idx],R2[idx],ideality_factor1[idx],Q1[idx],sigma[idx]])    

    return Zsum,np.array(Zparam)

def sim_cir4():
    """ Simulate circuit 4: R1 + (R2 || Q1) + ( (R3 + Z ) || Q2)"""
    R1=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr1= genZR(size_number,number_of_point,R1)

    R2=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr2= genZR(size_number,number_of_point,R2)

    ideality_factor1=np.round(lin_rand(alpha_range[0],alpha_range[1],size_number),3)
    Q1=log_rand(q_range[0],q_range[1],size_number)
    Zq1= genZQ(size_number,number_of_point,Q1,ideality_factor1,angular_frequency)

    ideality_factor2=np.round(lin_rand(alpha_range[0],alpha_range[1],size_number),3)
    Q2=log_rand(q_range[0],q_range[1],size_number)
    Zq2= genZQ(size_number,number_of_point,Q2,ideality_factor1,angular_frequency)

    R3=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr3= genZR(size_number,number_of_point,R3)

    sigma=log_rand(sigma_range[0],sigma_range[1],size_number)
    Zw=genZW(size_number,number_of_point,sigma,angular_frequency)
    
    Zsum=Zr1 + 1 / ( 1 / Zr2 + 1 / Zq1 ) + 1 / ( 1 / Zq2 + 1 / ( Zr3 + Zw ) )
    
    Zparam=[]
    for idx in range(size_number):
        Zparam.append([R1[idx],R2[idx],R3[idx],ideality_factor1[idx],Q1[idx],ideality_factor2[idx],Q2[idx],sigma[idx]])    

    return Zsum,np.array(Zparam)

def sim_cir5():
    """ Simulate circuit 5: R1 + ( R2 + ((R3 + Z) || Q2) ) || Q1 )"""
    R1=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr1= genZR(size_number,number_of_point,R1)

    R2=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr2= genZR(size_number,number_of_point,R2)

    ideality_factor1=np.round(lin_rand(alpha_range[0],alpha_range[1],size_number),3)
    Q1=log_rand(q_range[0],q_range[1],size_number)
    Zq1= genZQ(size_number,number_of_point,Q1,ideality_factor1,angular_frequency)

    R3=log_rand(resistance_range[0],resistance_range[1],size_number)
    Zr3= genZR(size_number,number_of_point,R3)

    ideality_factor2=np.round(lin_rand(alpha_range[0],alpha_range[1],size_number),3)
    Q2=log_rand(q_range[0],q_range[1],size_number)
    Zq2= genZQ(size_number,number_of_point,Q2,ideality_factor1,angular_frequency)

    sigma=log_rand(sigma_range[0],sigma_range[1],size_number)
    Zw=genZW(size_number,number_of_point,sigma,angular_frequency)
    
    Zsum=Zr1 + 1 / ( 1 / ( Zr2 +  1 / (( 1 / (Zr3 + Zw )) + 1 / Zq2 ) ) + 1 / Zq1 ) 
    
    Zparam=[]
    for idx in range(size_number):
        Zparam.append([R1[idx],R2[idx],R3[idx],ideality_factor1[idx],Q1[idx],ideality_factor2[idx],Q2[idx],sigma[idx]])    

    return Zsum,np.array(Zparam)

def sim_cir6(size_number=131072, number_of_point=60):
    """ Simulate circuit 6: L + R0 + (R1 || C1) + ( (R2 + Z ) || C2)"""
    angular_frequency = F_range(0.02,20000,number_of_point)[0]

    L = nor_rand(mu=3.4e-13, size_number=size_number)
    ZL = genZL(size_number, number_of_point, L)

    R0 = nor_rand(mu=0.4, size_number=size_number)
    ZR0 = genZR(size_number,number_of_point,R0)

    R1 = nor_rand(mu=0.85, size_number=size_number)
    ZR1 = genZR(size_number,number_of_point,R1)

    R2 = nor_rand(mu=0.5, size_number=size_number)
    ZR2 = genZR(size_number,number_of_point,R2)

    ideality_factor= np.ones(size_number)
    C1 = nor_rand(mu=0.04, size_number=size_number)
    ZC1= genZQ(size_number,number_of_point, C1,ideality_factor,angular_frequency)

    ideality_factor= np.ones(size_number)
    C2 = nor_rand(mu=0.0015, size_number=size_number)
    ZC2= genZQ(size_number,number_of_point, C2,ideality_factor,angular_frequency)

    sigma = nor_rand(mu=0.13, size_number=size_number)
    ZW=genZW(size_number,number_of_point,sigma,angular_frequency)

    Zsum= ZL + ZR0 + 1 / ( 1 / ZR1 + 1 / ZC1 ) + 1 / ( 1 / ZC2 + 1 / ( ZR2 + ZW ) )

    Zparam=[]
    for idx in range(size_number):
        Zparam.append([L[idx], R0[idx], R1[idx], R2[idx], C1[idx], C2[idx], sigma[idx]])    

    return Zsum,np.array(Zparam)


###### Initialize Parameters ######

#Number of circuit:
number_of_circuit=6    
#Number of spectrum in each circuit : 256 512 1024 2048 4096 8192 16384 32768 (131072)
size_number=131072
#Numer of data point in each spectrum:
number_of_point=60  
#Range of frequency:
angular_frequency=F_range(0.02,20000,number_of_point)[0] 
#Range of resistance:
resistance_range=[10**-1,10**4] 
#Range of idality factor of CPE:
alpha_range=[0.8,1]
#Range of CPE capacitance:
q_range=[10**-5,10**-3]
#Range of sigma:
sigma_range=[10**0,10**3]

Circuit_spec=np.zeros((number_of_circuit,size_number,number_of_point), dtype=complex)


Circuit_spec[0],Circuit0_param=sim_cir1()
Circuit_spec[1],Circuit1_param=sim_cir2()
Circuit_spec[2],Circuit2_param=sim_cir3()
Circuit_spec[3],Circuit3_param=sim_cir4()
Circuit_spec[4],Circuit4_param=sim_cir5()
Circuit_spec[5],Circuit5_param=sim_cir6()


param_dict={"Circuit0_param":Circuit0_param,"Circuit1_param":Circuit1_param,
            "Circuit2_param":Circuit2_param,"Circuit3_param":Circuit3_param,
            "Circuit4_param":Circuit4_param, "Circuit5_param":Circuit5_param}


##### Data Export #####
def arrange_data(Circuit,cir_class,size_number,number_of_point):
    label = cir_class
    imge = Circuit.imag
    phase = np.degrees(np.arctan(Circuit.imag/Circuit.real))
    mag = np.absolute(Circuit)

    x= np.zeros((size_number,3,number_of_point))
    y= np.zeros(size_number)
    
    for idx in range(size_number):
            y[idx] = cir_class
            for idx2 in range(number_of_point): 
                x[idx][0][idx2] = imge[idx][idx2]
                x[idx][1][idx2] = phase[idx][idx2]
                x[idx][2][idx2] = mag[idx][idx2]

    return x,y


def export_data(Circuit,size_number,number_of_point,numc):    
    x= np.zeros((numc,size_number,3,number_of_point))
    y= np.zeros((numc,size_number))
    
    for idx in range(numc):
        x[idx],y[idx]=arrange_data(Circuit[idx],(idx),size_number,number_of_point)
    
    x_data=x[0]
    y_data=y[0]
    
    for idx in range(numc-1):
        x_data=np.append(x_data,x[idx+1],axis=0)
        
    for idx in range(numc-1):
        y_data=np.append(y_data,y[idx+1],axis=0)
    
    return x_data,y_data