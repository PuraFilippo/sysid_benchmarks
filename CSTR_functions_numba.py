import random
import numpy as np
import copy
from scipy.integrate import solve_ivp, odeint
from numba import njit
from lti import f2, RK4_system
from odeintw import odeintw

def generate_input(rng, batch_size=1, N=1000, p_low_pass=0.5, p_high_pass=0.05):
    """
    Generate input signals with specified filtering characteristics.

    Parameters:
    batch_size (int): Number of signal batches to generate. Each batch contains one signal,
                      enabling the simultaneous generation of multiple signals.

    N (int): Total number of data points in each signal. This defines both the length of the
             signal in the time domain and the resolution in the frequency domain.

    p_low_pass (float): Probability (range 0 to 1) that the minimum frequency will be set to zero,
                        effectively applying a low-pass filter.

    p_high_pass (float): Probability (range 0 to 1) that the maximum frequency will be set to the
                         Nyquist frequency, effectively applying a high-pass filter.

    Returns:
    u (ndarray): Array of generated time-domain signals. The array shape is (batch_size, N),
                 where each row represents a single time-domain signal.

    uf (ndarray): Array of frequency-domain representations of the signals. This complex array
                  corresponds to the frequency components of the signals in 'u'.

    fmin (int): Minimum frequency index that is active in the frequency domain representation,
                defining the lower bound of the filter's passband.

    fmax (int): Maximum frequency index that is active, defining the upper bound of the filter's
                passband.
    """

    # Randomly select two frequencies within the valid range
    f1, f2 = rng.integers(size=2, low=1, high=N // 2 + 1)
    # Determine minimum and maximum frequencies based on random selection
    fmin, fmax = sorted([f1, f2])

    # Apply low-pass filter with the probability p_low_pass
    if rng.uniform() < p_low_pass:
        fmin = 1
    # Apply high-pass filter with the probability p_high_pass
    if rng.uniform() < p_high_pass:
        fmax = N // 2

    # Create an array of zeros for frequency components
    uf = np.zeros((batch_size, N // 2 + 1), dtype=np.complex64)
    # Assign random phase shifts to the frequency components within the passband
    uf[:, fmin:fmax + 1] = np.exp(1j * rng.uniform(low=0, high=2 * np.pi, size=(batch_size, fmax - fmin + 1)))

    # Inverse real FFT to convert frequency domain to time domain
    u = np.fft.irfft(uf)
    # Normalize the signal
    u *= np.sqrt((N // 2 + 1) / (fmax - fmin + 1) * N)

    return u, uf, fmin, fmax


def generate_random_binary_signal_rep(choices,min_repetitions, max_repetitions,length, input_RNG): 
    #INPUT RNG GOOD?
    '''
    It create a binary signal of dimension length picking the elements from 
    the list choices, constraining the signal to be formed by subsequences with
    the same element of length between min_repetitions and max_repetitions
    Args:
        choices: list of the 2 possible choices to pick the elements 
        min_repetitions: minimum number of repetitions in subsequence
        max_repetitions: maximum number of repetitions in subsequence
        length: total length of the signal
        input_RNG:
    Return:
        vec: the binary signal that satisfy the different conditions
    '''
    assert min_repetitions<=max_repetitions, "The minimum number of repetitions of the same value has to be smaller of the maximum one"
    assert len(choices) == 2, "We want only two possible choices, to create a binary signal"
    assert length>min_repetitions, "In order to not have infinite loops in the script"
    # chose randomly one element of choices, and repeat it in an appropriate way randomly
    rep = int(input_RNG.uniform(min_repetitions,max_repetitions))
    rep_vec = [rep]
    vec = []
    first_choice = [input_RNG.choice(choices)]
    vec = first_choice*rep
    #pick the other element in choices
    choice = [choices[np.where(np.array(first_choice*2)!=np.array(choices))[0][0]]]
    flag = True
    while flag:
        # compute how many times to repeat it and comput the remamining in order to not have problems with the dimension of length
        rep = int(input_RNG.uniform(min_repetitions,max_repetitions))
        remaining = length-len(vec)-rep
        if remaining>=min_repetitions and remaining<=max_repetitions:
            # if the remaining is between the min and max, you append rep in the vector, and continue the remaining with the other choice
            vec = np.concatenate([vec,choice*rep])
            rep_vec.append(rep)
            other_choice = [choices[np.where(np.array(choice*2)!=np.array(choices))[0][0]]]
            vec = np.concatenate([vec,other_choice*remaining])
            rep_vec.append(remaining)
            # print(rep_vec,sum(rep_vec))
            flag = False
        elif remaining>max_repetitions:
            # if is far from the ending, so you append normally in the vector
            rep_vec.append(rep)
            vec = np.concatenate([vec,choice*rep])
            first_choice = choice
            choice = [choices[np.where(np.array(first_choice*2)!=np.array(choices))[0][0]]]

        #if anything is not good you have to pick another number of ripetition until appropriate
    
    # print(vec[100:200])
    return vec


def CSTR_parameters(par_shifted,par_changed,par_fixed,seq_len, input_RNG):
    '''
    This taking the different parameters of the CSTR model it creates the 
    r.h.s. of the differential system using the different parameters taken as inputs
    in the structure of a system identification problem.
    I.E. Creating a explicit ODE, in which the input is a function of the time
    Args:
        par_shifted: parameters that will be shifted in the initial choice of their values
        changing them in percentage
        par_changed: parameters that will be affected by the shift in the shifted ones
        par_fixed: parameters that will be constant with respect to all the shifts 
        seq_len: the length of the sequence of the element taken as output
        input_RNG:
    Return:
        t_values: the timing values that are the correspective elements of the stepsize but in the 
        time formality
        u_values: the input values at each stepsize of the t_values
        y_values: the output values at each stepsize of the t_values

    '''
    # you chose a time that is 20 times of the number of steps in the numerical solution of the ODE
    t0,tN = 0,seq_len*20


    # parameters and the ones that comes from the par_fixed
    [E,lam,k0,Ca0,T0,TCin,UAJ] = par_shifted
    [kss,VR,D,AJ] = par_changed
    [R,rho,cp,U,rhoJ,cJ,Fss,TRss,Cass] = par_fixed


    #values from M., cite paper
    u1 = 11.26e-3
    choices = [0.6*u1,1.4*u1]

    # considering that return a vector, and the definition over the step jumps
    # is not well defined the only idea is to not build a function but 
    #directly a vector that return the same input values of the y
    FJss_seq = generate_random_binary_signal_rep(choices,20,80,seq_len, input_RNG)
    # FJss_seq = input_RNG.normal(size=(seq_len))*0.4*u1 +u1
    # FJss_seq = generate_input(batch_size=1, N=seq_len , p_low_pass=0.5, p_high_pass=0.05,
    #                     rng=input_RNG)[0].reshape(-1)*0.4*u1 +u1


    # signal = input_RNG.normal(size=(seq_len))
    # in_max = signal.max()
    # in_min = signal.min()
    # FJss_seq = choices[0] + ((signal - in_min) * (choices[1] - choices[0])) / (in_max - in_min)
    # def FJss(t):
    #     return generate_random_binary_function_rep(t,t0,tN,FJss_seq)


    # definition of the differentiation system


    # in order to not have problems with dimensions!
    #u_values = np.append(FJss_seq,FJss_seq[-1]) 
    u_values = FJss_seq[...,None]
    # numerical solution computed with RK4 and the plot of the function
    y0 = np.array([Ca0,T0,TCin])
    # print(f'i')
    t_values, y_values = RK4_system(f2,t0,y0,  tN, seq_len,FJss_seq,par_shifted,par_fixed,par_changed)

    return t_values,u_values, y_values[:-1]

# @njit

# @njit
# def f(t,y,FJss_seq,N,par_shifted,par_fixed,par_changed):



def MC_parameters_CSTR(seq_len, par_fixed,par_shifted,shift_RNG, input_RNG,shift = [0.03,0.1]):
    #SHIFT RNG used not only to choose the % of the shifts but also on the initial values
    '''
    It shift in an appropriate way the different parameters of the par_shifted list (using
    the values of the shift list) and run the CSTR system with all the appropriate parameters
    either shifted and changed wrt to the firs ones.
    Args:
        seq_len: the length of the sequence of the element taken as output
        par_fixed: parameters that will be constant with respect to all the shifts 
        par_shifted: parameters that will be shifted in the initial choice of their values
        changing them in percentage
        shift_RNG:
        input_RNG:
        shift: the maximum value of the percentage in which shift the different parameters
    Return:
        t_values: the timing values that are the correspective elements of the stepsize but in the 
        time formality
        FJss: the input values at each stepsize of the t_values
        y_values: the output values at each stepsize of the t_values
    '''
    # considering the structure of the ODE, it is trivial to notice that the influence of 
    # the totalenergy on the numerical solution is particularly strong, i.e. it is possible
    # to see in the ODE a exponential dependence wrt to all the other parameters that have
    # a linear one, for this reason it is fixed the shift for the energy t 3%, while the 
    # other can be chosen by the user but it should be less than 20%
    
    [R,rho,cp,U,rhoJ,cJ,Fss,TRss,Cass] = par_fixed
    par_shifted = copy.copy(par_shifted)
    [shift_E,shift_par] = shift 
    assert shift_E<=0.03, "The change in percentage on the E should not exceed 3%"
    assert shift_par<=0.2, "The change in percentage on the param should not exceed 20%"

    #shift the ones with max 10%
    par_shifted[1] = shift_RNG.uniform((1+shift_par)*par_shifted[1],(1-shift_par)*par_shifted[1])
    par_shifted[2] = shift_RNG.uniform((1-shift_par)*par_shifted[2],(1+shift_par)*par_shifted[2])
    # shift the energy that depends exponentiallly in the ODE so at maximum of 3%
    par_shifted[0] = shift_RNG.uniform((1-shift_E)*par_shifted[0],(1)*par_shifted[0])

    # Set the initial values more similar to the one in steadystate, and they are connected to each other in order to have a 
    # steady state of TR similar to 350, the nominal one, because if both chosen randomly this characteristic is more changable
    random_Ca = shift_RNG.uniform(0.3,0.7)
    random_T = 10*random_Ca/0.7
    par_shifted[3] = par_shifted[3] + random_Ca
    par_shifted[4] = par_shifted[4] -random_T
    par_shifted[5] = par_shifted[5] -random_T
    
    # how the shifts in the parameters affect the par_changed
    kss = par_shifted[2]*np.exp(-par_shifted[0]/((TRss*R)))
    VR = (Fss*(par_shifted[3] - Cass)/(kss*Cass))
    D = (2*VR/np.pi)**(1/3)
    AJ = 2*np.pi*D**2
    UAJ = shift_RNG.uniform((1-shift_par)*U*AJ,(1+shift_par)*U*AJ)
    # concatenate the one that is not only changed bu also shifted
    par_shifted = np.concatenate([par_shifted,[UAJ]])
    par_changed = [kss,VR,D,AJ]
    # print(f'changed{par_changed}')
    # print(f'shift{par_shifted}')
    
    par_fixed = [np.float64(x) for x in par_fixed]
    t_values,FJss, y_values = CSTR_parameters(par_shifted,par_changed,par_fixed,seq_len, input_RNG)
    # print(y_values)
    
    return t_values,FJss,  y_values