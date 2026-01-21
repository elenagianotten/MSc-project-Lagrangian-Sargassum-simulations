#This file contains the kernels with additional transport mechanisms acting on Sargassum 
#Stokes and windage kernels are taken from Darshika Manral's project: https://github.com/OceanParcels/woc_sargassum_transport/blob/depth_avg/src/simulations/custom_kernels.py

import math

#Simple implementation of Stokes drift
def direct_Stokes_drift(particle, fieldset, time):
    # Sample the U / V components of Stokes drift
    stokes_U = fieldset.U_wave_Stokes[time, particle.depth, particle.lat, particle.lon]
    stokes_V = fieldset.V_wave_Stokes[time, particle.depth, particle.lat, particle.lon]

    # compute particle displacement
    particle_dlon += stokes_U * particle.dt
    particle_dlat += stokes_V * particle.dt 


def avg_Stokes_drift(particle, fieldset, time):
    """Stokes drift kernel: taken from https://github.com/OceanParcels/plasticparcels/kernels.py

    Description
    ----------
    Using the approach in [1] assuming a Phillips wave spectrum to determine
    the depth dependent Stokes drift. Specifically, the 'Stokes drift velocity'
    :math:`u_s` is computed as per Eq. (19) in [1].

    We treat the Stokes drift as a linear addition to the velocity field
        :math:`u(x,t) = u_c(x,t) + C_s * u_s(x,t)`
    where :math:`u_c` is the current velocity, :math:`u_s` is the Stokes drift velocity,
    and :math:`C_s` is the depth-varying decay factor.

    For further description, see https://plastic.oceanparcels.org/en/latest/physicskernels.html#stokes-drift

    Parameter Requirements
    ----------
    fieldset :
        - `fieldset.Stokes_U` and `fieldset.Stokes_V`, the Stokes drift velocity fields. Units [m s-1]
        - `fieldset.wave_Tp`, the peak wave period field (:math:`T_p`). Units [s].

    References
    ----------
    [1] Breivik (2016) - https://doi.org/10.1016/j.ocemod.2016.01.005

    """
    # Sample the U / V components of Stokes drift
    stokes_U = fieldset.U_wave_Stokes[time, particle.depth, particle.lat, particle.lon]
    stokes_V = fieldset.V_wave_Stokes[time, particle.depth, particle.lat, particle.lon]

    # Sample the peak wave period
    T_p = fieldset.wave_Tp[time, particle.depth, particle.lat, particle.lon]

    # Compute the local bathymetry / water depth with a margin of error
    # local_bathymetry = 0.99*fieldset.bathymetry[time, particle.depth, particle.lat, particle.lon]

    # Only compute displacements if the peak wave period is large enough and the particle is in the water
    if T_p > 1E-14: #and particle.depth < local_bathymetry:
        # Peak wave frequency
        omega_p = 2. * math.pi / T_p

        # Peak wave number
        k_p = (omega_p ** 2) / fieldset.G

        particle.k_p = k_p

        # Repeated inner term of Eq. (19) - note depth is negative in this formulation, but model depths are positive by convention
        # kp_z_2 = 2. * k_p * particle.depth
        kp_z_2 = 2. * k_p * particle.depth_extent / 2


        # Decay factor in Eq. (19) -- Where beta=1 for the Phillips spectrum
        decay = math.exp(-kp_z_2) - math.sqrt(math.pi * kp_z_2) * math.erfc(math.sqrt(kp_z_2))

        # Saving decay function as particle variable
        particle.decay_averaged = decay

        # Apply Eq. (19) and compute particle displacement
        particle_dlon += stokes_U * decay * particle.dt  # noqa
        particle_dlat += stokes_V * decay * particle.dt  # noqa


def di_Stokes_drift(particle, fieldset, time):
    """Depth-integrated Stokes drift kernel:

    Description
    ----------
    Using the approach in [1] (assuming a Phillips wave spectrum), equation A.6 and A.7 of [2] are used to determine
    the Stokes drift velocity integrated over depth between upper extent and lower extent of particle. 

    Stokes drift is treated as a linear addition to the velocity field. 

    Parameter Requirements
    ----------
    fieldset :
        - fieldset.U_wave_Stokes: zonal Stokes drift velocity at surface [m s-1]
        - fieldset.V_wave_Stokes: meridional Stokes drift velocity at surface [m s-1]
        - fieldset.wave_Tp: the peak wave period field [s].

    References
    ----------
    [1] Breivik (2016) - https://doi.org/10.1016/j.ocemod.2016.01.005
    [2] Li et al. (2017) - http://dx.doi.org/10.1016/j.ocemod.2017.03.016  
    """

    delta_z = particle.depth_extent - particle.depth
    z_up = particle.depth
    z_low = z_up + particle.depth_extent

    #Sampling the U / V components of Stokes drift at upper level
    stokes_U = fieldset.U_wave_Stokes[time, particle.depth, particle.lat, particle.lon]
    stokes_V = fieldset.V_wave_Stokes[time, particle.depth, particle.lat, particle.lon]

    #Sampling the peak wave period
    T_p = fieldset.wave_Tp[time, particle.depth, particle.lat, particle.lon]

    #Only computing displacements if the peak wave period is large enough and the particle is in the water
    if T_p > 1E-14: #and particle.depth < local_bathymetry:
        #Peak wave frequency
        omega_p = 2. * math.pi / T_p

        #Peak wave number
        k_p = (omega_p ** 2) / fieldset.G

        #Decay function lower extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_lower = 1/(2*k_p) * ( 
                    1 - math.exp(-2.0*k_p*z_low) 
                    - (2.0/3.0) * (1 + math.sqrt(math.pi) * (2.0*k_p*z_low)**(3.0/2.0) * math.erfc(math.sqrt(2.0*k_p*z_low))  
                    - (1 + 2.0*k_p*z_low) * math.exp(-2.0*k_p*z_low)   )
                    )
        

        #Decay function upper extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_upper = 1/(2*k_p) * ( 
                    1 - math.exp(-2.0*k_p*z_up) 
                    - (2.0/3.0) * (1 + math.sqrt(math.pi) * (2.0*k_p*z_up)**(3.0/2.0) * math.erfc(math.sqrt(2.0*k_p*z_up))  
                    - (1 + 2.0*k_p*z_up) * math.exp(-2.0*k_p*z_up)   )
                    )
        
        #Integration function between surface and lower level based on Equation A.7 of Li et al. (2017)
        stokes_U_integrated = (stokes_U * decay_function_lower - stokes_U * decay_function_upper) / delta_z
        stokes_V_integrated = (stokes_V * decay_function_lower - stokes_V * decay_function_upper) / delta_z

        #Saving lower and upper decay function and total Stokes decay factor as particle variables
        particle.decay_integrated_lower = decay_function_lower
        particle.decay_integrated_upper = decay_function_upper
        particle.decay_factor = (decay_function_lower - decay_function_upper) / delta_z

        #Compute particle displacement based on depth-integrated Stokes velocity
        particle_dlon += stokes_U_integrated  * particle.dt  
        particle_dlat += stokes_V_integrated  * particle.dt 



def di_Stokes_drift_biomass_extent_dependency(particle, fieldset, time):
    """Depth-integrated Stokes drift kernel, with variable depth extent:

    Description
    ----------
    Using the approach in [1] (assuming a Phillips wave spectrum), equation A.6 and A.7 of [2] are used to determine
    the Stokes drift velocity integrated over depth between upper extent and lower extent of particle. 

    Stokes drift is treated as a linear addition to the velocity field. 

    Parameter Requirements
    ----------
    fieldset :
        - fieldset.U_wave_Stokes: zonal Stokes drift velocity at surface [m s-1]
        - fieldset.V_wave_Stokes: meridional Stokes drift velocity at surface [m s-1]
        - fieldset.wave_Tp: the peak wave period field [s].

    References
    ----------
    [1] Breivik (2016) - https://doi.org/10.1016/j.ocemod.2016.01.005
    [2] Li et al. (2017) - http://dx.doi.org/10.1016/j.ocemod.2017.03.016  
    """

    #Depth extent dependent on relative biomass
    particle.depth_extent = 0.25 + 0.25 * particle.biomass_SF3

    #Maximum depth extent is 1 m
    if particle.depth_extent > 1:
        particle.depth_extent = 1

    delta_z = particle.depth_extent - particle.depth
    z_up = particle.depth
    z_low = z_up + particle.depth_extent

    #Sampling the U / V components of Stokes drift at upper level
    stokes_U = fieldset.U_wave_Stokes[time, particle.depth, particle.lat, particle.lon]
    stokes_V = fieldset.V_wave_Stokes[time, particle.depth, particle.lat, particle.lon]

    #Sampling the peak wave period
    T_p = fieldset.wave_Tp[time, particle.depth, particle.lat, particle.lon]

    #Only computing displacements if the peak wave period is large enough and the particle is in the water
    if T_p > 1E-14: #and particle.depth < local_bathymetry:
        #Peak wave frequency
        omega_p = 2. * math.pi / T_p

        #Peak wave number
        k_p = (omega_p ** 2) / fieldset.G

        #Decay function lower extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_lower = 1/(2*k_p) * ( 
                    1 - math.exp(-2.0*k_p*z_low) 
                    - (2.0/3.0) * (1 + math.sqrt(math.pi) * (2.0*k_p*z_low)**(3.0/2.0) * math.erfc(math.sqrt(2.0*k_p*z_low))  
                    - (1 + 2.0*k_p*z_low) * math.exp(-2.0*k_p*z_low)   )
                    )
        

        #Decay function upper extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_upper = 1/(2*k_p) * ( 
                    1 - math.exp(-2.0*k_p*z_up) 
                    - (2.0/3.0) * (1 + math.sqrt(math.pi) * (2.0*k_p*z_up)**(3.0/2.0) * math.erfc(math.sqrt(2.0*k_p*z_up))  
                    - (1 + 2.0*k_p*z_up) * math.exp(-2.0*k_p*z_up)   )
                    )
        
        #Integration function between surface and lower level based on Equation A.7 of Li et al. (2017)
        stokes_U_integrated = (stokes_U * decay_function_lower - stokes_U * decay_function_upper) / delta_z
        stokes_V_integrated = (stokes_V * decay_function_lower - stokes_V * decay_function_upper) / delta_z

        #Saving lower and upper decay function and total Stokes decay factor as particle variables
        particle.decay_integrated_lower = decay_function_lower
        particle.decay_integrated_upper = decay_function_upper
        particle.decay_factor = (decay_function_lower - decay_function_upper) / delta_z

        #Compute particle displacement based on depth-integrated Stokes velocity
        particle_dlon += stokes_U_integrated  * particle.dt  
        particle_dlat += stokes_V_integrated  * particle.dt 



def di_Stokes_drift_biomass_extent_dependency(particle, fieldset, time):
    """Depth-integrated Stokes drift kernel, with variable depth extent:

    Description
    ----------
    Using the approach in [1] (assuming a Phillips wave spectrum), equation A.6 and A.7 of [2] are used to determine
    the Stokes drift velocity integrated over depth between upper extent and lower extent of particle. 

    Stokes drift is treated as a linear addition to the velocity field. 

    Parameter Requirements
    ----------
    fieldset :
        - fieldset.U_wave_Stokes: zonal Stokes drift velocity at surface [m s-1]
        - fieldset.V_wave_Stokes: meridional Stokes drift velocity at surface [m s-1]
        - fieldset.wave_Tp: the peak wave period field [s].

    References
    ----------
    [1] Breivik (2016) - https://doi.org/10.1016/j.ocemod.2016.01.005
    [2] Li et al. (2017) - http://dx.doi.org/10.1016/j.ocemod.2017.03.016  
    """
    #Depth extent dependent on relative biomass
    particle.depth_extent = 0.25 + 0.25 * particle.biomass_SF3

    #Maximum depth extent is 1 m
    if particle.depth_extent > 1:
        particle.depth_extent = 1

    delta_z = particle.depth_extent - particle.depth
    z_up = particle.depth
    z_low = z_up + particle.depth_extent

    #Sampling the U / V components of Stokes drift at upper level
    stokes_U = fieldset.U_wave_Stokes[time, particle.depth, particle.lat, particle.lon]
    stokes_V = fieldset.V_wave_Stokes[time, particle.depth, particle.lat, particle.lon]

    #Sampling the peak wave period
    T_p = fieldset.wave_Tp[time, particle.depth, particle.lat, particle.lon]

    #Only computing displacements if the peak wave period is large enough and the particle is in the water
    if T_p > 1E-14: #and particle.depth < local_bathymetry:
        #Peak wave frequency
        omega_p = 2. * math.pi / T_p

        #Peak wave number
        k_p = (omega_p ** 2) / fieldset.G

        #Decay function lower extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_lower = 1/(2*k_p) * ( 
                    1 - math.exp(-2.0*k_p*z_low) 
                    - (2.0/3.0) * (1 + math.sqrt(math.pi) * (2.0*k_p*z_low)**(3.0/2.0) * math.erfc(math.sqrt(2.0*k_p*z_low))  
                    - (1 + 2.0*k_p*z_low) * math.exp(-2.0*k_p*z_low)   )
                    )
        

        #Decay function upper extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_upper = 1/(2*k_p) * ( 
                    1 - math.exp(-2.0*k_p*z_up) 
                    - (2.0/3.0) * (1 + math.sqrt(math.pi) * (2.0*k_p*z_up)**(3.0/2.0) * math.erfc(math.sqrt(2.0*k_p*z_up))  
                    - (1 + 2.0*k_p*z_up) * math.exp(-2.0*k_p*z_up)   )
                    )
        
        #Integration function between surface and lower level based on Equation A.7 of Li et al. (2017)
        stokes_U_integrated = (stokes_U * decay_function_lower - stokes_U * decay_function_upper) / delta_z
        stokes_V_integrated = (stokes_V * decay_function_lower - stokes_V * decay_function_upper) / delta_z

        #Saving lower and upper decay function and total Stokes decay factor as particle variables
        particle.decay_integrated_lower = decay_function_lower
        particle.decay_integrated_upper = decay_function_upper
        particle.decay_factor = (decay_function_lower - decay_function_upper) / delta_z

        #Compute particle displacement based on depth-integrated Stokes velocity
        particle_dlon += stokes_U_integrated  * particle.dt  
        particle_dlat += stokes_V_integrated  * particle.dt 
        

def di_Stokes_drift_wave_damping(particle, fieldset, time):
    """Depth-integrated Stokes drift kernel, with variable depth extent:

    Description
    ----------
    Using the approach in [1] (assuming a Phillips wave spectrum), equation A.6 and A.7 of [2] are used to determine
    the Stokes drift velocity integrated over depth between upper extent and lower extent of particle. 

    Stokes drift is treated as a linear addition to the velocity field. 

    Parameter Requirements
    ----------
    fieldset :
        - fieldset.U_wave_Stokes: zonal Stokes drift velocity at surface [m s-1]
        - fieldset.V_wave_Stokes: meridional Stokes drift velocity at surface [m s-1]
        - fieldset.wave_Tp: the peak wave period field [s].

    References
    ----------
    [1] Breivik (2016) - https://doi.org/10.1016/j.ocemod.2016.01.005
    [2] Li et al. (2017) - http://dx.doi.org/10.1016/j.ocemod.2017.03.016  
    """

    delta_z = particle.depth_extent - particle.depth
    z_up = particle.depth
    z_low = z_up + particle.depth_extent

    #Sampling the U / V components of Stokes drift at upper level
    stokes_U = fieldset.U_wave_Stokes[time, particle.depth, particle.lat, particle.lon]
    stokes_V = fieldset.V_wave_Stokes[time, particle.depth, particle.lat, particle.lon]

    #Sampling the peak wave period
    T_p = fieldset.wave_Tp[time, particle.depth, particle.lat, particle.lon]

    #Only computing displacements if the peak wave period is large enough and the particle is in the water
    if T_p > 1E-14: #and particle.depth < local_bathymetry:
        #Peak wave frequency
        omega_p = 2. * math.pi / T_p

        #Peak wave number
        k_p = (omega_p ** 2) / fieldset.G

        #Decay function lower extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_lower = 1/(2*k_p) * ( 
                    1 - math.exp(-2.0*k_p*z_low) 
                    - (2.0/3.0) * (1 + math.sqrt(math.pi) * (2.0*k_p*z_low)**(3.0/2.0) * math.erfc(math.sqrt(2.0*k_p*z_low))  
                    - (1 + 2.0*k_p*z_low) * math.exp(-2.0*k_p*z_low)   )
                    )
        

        #Decay function upper extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_upper = 1/(2*k_p) * ( 
                    1 - math.exp(-2.0*k_p*z_up) 
                    - (2.0/3.0) * (1 + math.sqrt(math.pi) * (2.0*k_p*z_up)**(3.0/2.0) * math.erfc(math.sqrt(2.0*k_p*z_up))  
                    - (1 + 2.0*k_p*z_up) * math.exp(-2.0*k_p*z_up)   )
                    )
        
        #Integration function between surface and lower level based on Equation A.7 of Li et al. (2017)
        stokes_U_integrated = (stokes_U * decay_function_lower - stokes_U * decay_function_upper) / delta_z
        stokes_V_integrated = (stokes_V * decay_function_lower - stokes_V * decay_function_upper) / delta_z

        

        # #Calculate damping factor
        # damping_factor = 0.5 + 0.5 * math.cos(math.pi * (particle.biomass_SF3 - 1) / 3) #+ 0.25*x
        # if particle.biomass_SF3 < 1:
        #     damping_factor = 1.0
        # if particle.biomass_SF3 > 4:
        #     damping_factor = 0.0

        #Adding wave damping by linear damping factor acting on Stokes drift 
        linear_wave_damping_factor = 1.5 - 0.5*particle.biomass_SF3
        if particle.biomass_SF3 < 1:
            linear_wave_damping_factor = 1.0
        if particle.biomass_SF3 > 3:
            linear_wave_damping_factor = 0.0

        #Saving lower and upper decay function and total Stokes decay factor as particle variables
        particle.decay_integrated_lower = decay_function_lower
        particle.decay_integrated_upper = decay_function_upper
        particle.decay_factor = linear_wave_damping_factor * (decay_function_lower - decay_function_upper) / delta_z

        #Calculate damped Stokes drift based on a biomass dependent wave damping factor
        stokes_U_int_damped = linear_wave_damping_factor * stokes_U_integrated
        stokes_V_int_damped = linear_wave_damping_factor * stokes_V_integrated    

        #Compute particle displacement based on depth-integrated Stokes velocity
        particle_dlon += stokes_U_int_damped  * particle.dt  
        particle_dlat += stokes_V_int_damped  * particle.dt 

def wind_drag(particle, fieldset, time):

    (curr_U, curr_V) = fieldset.UV[particle]
    ocean_speed = math.sqrt(curr_U**2 + curr_V**2)

    if ocean_speed > 1E-14:
        # Sample the U / V components of wind
        wind_U = fieldset.U_wind[time, particle.depth, particle.lat, particle.lon]
        wind_V = fieldset.V_wind[time, particle.depth, particle.lat, particle.lon]

        # compute particle displacement
        particle_dlon += wind_U* particle.dt
        particle_dlat += wind_V * particle.dt 

def windage_drift(particle, fieldset, time):
    """Leeway windage kernel.

    Description
    ----------
    A simple windage kernel that applies a linear relative 'wind velocity'
    to the particle. Slightly adapted for the usage for Sargassum.

    We treat the windage drift as a linear addition to the velocity field
        :math:`u(x,t) = u_c(x,t) + C_w * (u_w(x,t)-u_c(x,t))`
    where :math:`u_c` is the ocean current velocity, :math:`u_w` is the wind velocity
    at 10m height, and :math:`C_w` is the windage coefficient (usually taken to
    be in [1%,5%], depending on particle size)

    For further description, see https://plastic.parcels-code.org/en/latest/physicskernels.html#wind-induced-drift-leeway

    Parameter Requirements
    ----------
    particle :
        - wind_coefficient - the particle windage coefficient in decimals.
    fieldset :
        - `fieldset.Wind_U` and `fieldset.Wind_V`, the wind velocity field at 10m height above sea surface. Units [m s-1].

    Kernel Requirements
    ----------
    Order of Operations:
        None - can be applied at any time.

    """
    # Sample ocean velocities
    (ocean_U, ocean_V) = fieldset.UV[particle]
    ocean_speed = math.sqrt(ocean_U**2 + ocean_V**2)

    # Use a basic approach to only apply windage to particle in the ocean
    if ocean_speed > 1E-14:
        # Sample the U / V components of wind
        wind_U = fieldset.U_wind[time, particle.depth, particle.lat, particle.lon]
        wind_V = fieldset.V_wind[time, particle.depth, particle.lat, particle.lon]

        # Compute particle displacement
        particle_dlon += particle.wind_coefficient * (wind_U - ocean_U) * particle.dt  # noqa
        particle_dlat += particle.wind_coefficient * (wind_V - ocean_V) * particle.dt  # noqa



#Kernel that samples temperature field, salinity field and nitrogen field at particle location
def sampling_from_field(particle, fieldset, time):
    particle.temperature = fieldset.T[time, particle.depth, particle.lat, particle.lon]
    particle.salinity = fieldset.S[time, particle.depth, particle.lat, particle.lon]
    
    #Selecting depth at which nitrogen field is defined
    z_for_n = particle.depth
    if z_for_n <= 0.49402538:
        z_for_n = 0.49402538
    
    particle.nitrogen = fieldset.no3[time, z_for_n, particle.lat, particle.lon] 



#Kernel that determines the new weight of the particle 
#Based on the maximum growth rates of morphotypes and multiple limitation curves
def sargassum_biological_growth_model(particle, fieldset, time): 

    #Maximum growth rate measured by Magana-Gallegos et al. (2023)
    MGR_SF3_MG = 0.095 #[Doubling/day]
    MGR_SN8_MG = 0.059 #[Doubling/day]
    MGR_SN1_MG = 0.063 #[Doubling/day]

    #GROWTH LIMITATION FUNCTION DEPENDENT ON TEMPERATURE
    #Sargassum model parameters based on Jouanno et al. (2025)
    T_opt_J25 = 27.5    #Temperature growth optimum [degC]
    Tmin_J25 = 20       #Temperature growth minimum [degC]
    Tmax_J25 = 31       #Temperature growth maximum [degC]
    
    #Formula from Jouanno et al. (2025).
    if particle.temperature < T_opt_J25:
        limitation_factor_T = math.exp(-2 * ( (particle.temperature - T_opt_J25)/ (Tmin_J25 - T_opt_J25))**2 )
    else:
        limitation_factor_T = math.exp(-2 * ( (particle.temperature - T_opt_J25)/ (Tmax_J25 - T_opt_J25))**2 )

    #GROWTH LIMITATION FUNCTION DEPENDENT ON NITROGEN AVAILABILTIY
    #Model parameters from Bonner et al. (2024).
    k_N = 0.001 #Nitrogen uptake half saturation [mmol/m3]

    #Formula from Bonner et al. (2024)
    limitation_factor_N = particle.nitrogen / ( k_N + particle.nitrogen )

    #GROWTH LIMITATION FUNCTION DEPENDENT ON SALINITY
    #Sargassum model parameters based on Jouanno et al. (2025)
    S_opt_J25 = 36.0      #psu

    #Formula from Jouanno et al. (2025)
    limitation_factor_S = math.exp(-0.02 * (S_opt_J25 - particle.salinity)**2 )

    ###################################

    #Save particle total limitation as a 
    LIMITATION = limitation_factor_T * limitation_factor_N * limitation_factor_S
    particle.limitation = LIMITATION
    particle.lim_salinity = limitation_factor_S
    particle.lim_temp = limitation_factor_T
    particle.lim_no3 = limitation_factor_N
    
    #Mortality
    mortality1 = 0.025 #day-1
    mortality2 = 0.04

    #UPDATE OF PARTICLE WEIGHT with doubling rate and mortality rate converted from day-1 to s-1

    if particle.stranded == 1 :
        particle.biomass_SF3 = particle.biomass_SF3

    else:
        particle.biomass_SF3 *= 2 ** ((LIMITATION * (MGR_SF3_MG / (24*60*60)) - mortality1 / (24*60*60) ) * particle.dt ) 
        particle.biomass_SN8 *= 2 ** ((LIMITATION * (MGR_SF3_MG / (24*60*60)) - mortality2 / (24*60*60) ) * particle.dt ) 
        particle.biomass_SN1 *= 2 ** (LIMITATION * (MGR_SF3_MG / (24*60*60))  * particle.dt ) 
        particle.biomass_loss = particle.biomass_SN1 - particle.biomass_SF3


def stranding(particle, fieldset, time): 
    u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    v = fieldset.V[time, particle.depth, particle.lat, particle.lon]

    if u == 0.0 or v == 0.0:
        particle.stranded = 1
    
    if particle.stranded == 1:
        particle_dlon = 0.0
        particle_dlat = 0.0
        #dlon dlat naar 0 zetten

    if particle.lat > 33.0:
        particle_dlat = 0.0
        particle_dlon = 0.0

def velocity_contribution(particle, fieldset, time):
    #Currents (applying unit converter from lon/lat s-1 to m s-1)
    currents_U = fieldset.U.eval(time, particle.depth, particle.lat, particle.lon, applyConversion=False) 
    currents_V = fieldset.V.eval(time, particle.depth, particle.lat, particle.lon, applyConversion=False)
    particle.speed_currents = math.sqrt(currents_U**2 + currents_V**2)

    #Wind (applying unit converter from lon/lat s-1 to m s-1)
    wind_U = fieldset.U_wind.eval(time, particle.depth, particle.lat, particle.lon, applyConversion=False)
    wind_V = fieldset.V_wind.eval(time, particle.depth, particle.lat, particle.lon, applyConversion=False)
    particle.speed_wind = math.sqrt(wind_U**2 + wind_V**2)

    #Stokes (applying unit converter from lon/lat s-1 to m s-1)
    #REMEMBER THAT THIS IS SURFACE STOKES DRIFT SO USE DECAY FACTOR (STOKES DI KERNEL) IN POSTPROCESSING
    stokes_U = fieldset.U_wave_Stokes.eval(time, particle.depth, particle.lat, particle.lon, applyConversion=False)
    stokes_V = fieldset.V_wave_Stokes.eval(time, particle.depth, particle.lat, particle.lon, applyConversion=False)
    particle.speed_stokes = math.sqrt(stokes_U**2 + stokes_V**2)


