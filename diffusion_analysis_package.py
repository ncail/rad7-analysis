# Author: Austin Helgert and Joseph Street
# Description: Code modules that are used for the diffusion analyses.


# class ClassName(object):

# 	"""docstring for ClassName"""
# 	def __init__(self, arg):
# 		super(ClassName, self).__init__() # ???

# 		self.MeasurementName = arg
		

# ************************************
# Imports
# ************************************

# System
import sys
import os

# Math
import numpy as np
import scipy
from scipy import stats
from scipy.optimize import curve_fit, least_squares, leastsq
from scipy.integrate import solve_ivp

# Plotting
import matplotlib.pyplot as plt
import plotly.express as px

# RAD7
import Rad7_API




def getData(RAD7_name='SDSMT', fileName='231218_diff_hotSide_leakCheck', binSize_hours=1/3, onlyPo218=True):

	sys.path.append('../../pythonCode/')

	# path and filename of data
	dataPath = '../../data/'+RAD7_name+'/'
	fileList = [fileName+'.r7raw']

	# process raw data and return DataFrame
	df = Rad7_API.GetProcessedRuns(dataPath, fileList=fileList, binHours=binSize_hours, onlyPo218=onlyPo218)

	return df

def placeCut(df, t_i='2023-12-15 00:00:00', t_f='2023-12-16 06:00:00'):

	sys.path.append('../../pythonCode/')

	# select a part of df
	dfc = Rad7_API.placeTimeCut(df, t_i, t_f)

	# add (or zero) Days and Seconds columns
	dfc = Rad7_API.addTimeColumns(dfc)

	return dfc


def goodnessOfFit(observed, expected, sigma, numberOfFittedParams):
	
	ndf = len(observed)-numberOfFittedParams

	if type(observed)==list:
		print("'observed' was a list, converting to array...")
		observed = np.asarray(observed)
	if type(expected)==list:
		print("'expected' was a list, converting to array...")
		expected = np.asarray(expected)

	chi_squared = (((observed-expected)**2)/sigma**2).sum()

	reduced_chi_squared = chi_squared/ndf

	p_value = 1 - stats.chi2.cdf(x=chi_squared, df=ndf) # find p-value
	# NOTICE: The p-value will likely be garbage because "expected" is used
	# instead of a true sigma^2 in calculating the chi-squared.

	return chi_squared, ndf, reduced_chi_squared, p_value


def exponentialFit(t, amp, t_char, baseline):

	y = amp * np.exp(-t/t_char) + baseline

	return y


def exponentialFit_zeroBaseline(t, amp, t_char): return exponentialFit(t, amp, t_char, 0)



def performFit(data_t, data_y, data_u, p0, bounds, fitting_function=exponentialFit):

	popt, pcov = curve_fit(fitting_function, xdata=data_t, ydata=data_y, sigma=data_u, p0=p0, bounds=bounds)

	uncerts = np.sqrt(np.diag(pcov))
	fit = fitting_function(data_t, *popt)

	gof = goodnessOfFit(data_y, fit, data_u, len(popt))

	print(f'\npopt = {popt} +/- {uncerts}\npcov = {pcov}')
	print(f'chi squared / ndf = {gof[0]:.2f} / {gof[1]:.0f} = {gof[0] / gof[1]:.2f}, p-value = {gof[3]:.3e}')

	# get all variables defined in this function so that we may return them
	local_variables = locals()

	return local_variables


def getPlot(df, dfc=None, in_days=False, save_plot_as=None, savePath='Plots/', measurementName='dummyMeasurement', extra_curves=None, xLimits=None, yLimits=None, yAxis=None):

	# get binSize_hours:
	binSize_hours = df.Hours.iloc[1] - df.Hours.iloc[0]

	# are we plotting in days or datetime?
	time = df['DateTime']
	if in_days == True: # we are plotting in days!
		time = df['Days']

	# create figure
	plt.figure(figsize=(20,8))

	# plot df
	plt.errorbar(time, df.RnConc, df.Uncert_RnConc, fmt='o', color = 'k', lw=3, capsize=10, alpha = 1, capthick=3, label='All data')


	# plotting selected region of df (i.e., dfc) that will be used for fitting?
	if dfc is not None:

		if in_days == True: # we are plotting in days!

			# We want to shift dfc to be consistent with df.
			# Get time between start of df and dfc in days:
			timeShift = (dfc['DateTime'].iloc[0] - df['DateTime'].iloc[0]).total_seconds() / 86400

			plt.errorbar(dfc.Days + timeShift, dfc.RnConc, dfc.Uncert_RnConc, fmt='o', color = 'dodgerblue', lw=3, capsize=10, alpha = 1, capthick=3, label='Selected data')


			if extra_curves is not None:
				for extra_curve in extra_curves:

					x, y = extra_curve['x'], extra_curve['y']
					
					plot_meta = extra_curve.copy()
					for k in ['x', 'y']:
						plot_meta.pop(k)
					
					plt.plot(x + timeShift, y, **plot_meta, zorder=10)


		else: # we are plotting in datetime!
			plt.errorbar(dfc.DateTime, dfc.RnConc, dfc.Uncert_RnConc, fmt='o', color = 'dodgerblue', lw=3, capsize=10, alpha = 1, capthick=3, label='Selected data')

			if extra_curves is not None:
				print(f'CAUTION: extra_curve is only plotting over Days (not DateTime).')


	# set axis labels
	plt.ylabel(r'Radon Concentration [Bq/m$^3$] with '+str(int(round(binSize_hours*60,2)))+'-min Bins', fontsize = 20)
	if in_days == True:
		plt.xlabel(r'Time [days]', fontsize = 20)

	plt.legend(fontsize = 18)

	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)

	if xLimits is not None:
		plt.xlim(xLimits[0], xLimits[1])

	if yLimits is not None:
		plt.ylim(yLimits[0], yLimits[1])

	if yAxis is not None:
		plt.yscale("log")

	plt.grid()
	plt.tight_layout()

	if save_plot_as is not None:

		try:
			os.mkdir(savePath+measurementName)
		except:
			print('Measurement folder already exists.')

		savePath_and_name = savePath+measurementName+'/'+save_plot_as+'.pdf'

		plt.savefig(savePath_and_name, format='pdf', dpi=300, bbox_inches='tight')
		print('plot saved: '+savePath_and_name)

	plt.show()


def getPlot_interactive(df,  xColumn='Days', yColumn='RnConc', errorColumn='Uncert_RnConc'):

	fig = px.scatter(df, x=xColumn, y=yColumn,error_y=errorColumn)

	fig.show()



###################											 ###################
###################			  NUMERICAL SOLUTION             ###################
###################											 ###################


# DEFINE DIFFERENTIAL EQUATIONS AND MINIMIZER

def model_full(vs, minimize_model=False, D_val=1e-13, P_val=1e-13, tau_cold_val=5.516, p0=None, bounds=None, return_expected=False, returnGOF=False, verbose=False):

	# Numerical Solution Technical Parameters
	total_time = vs.dfc.Days.iloc[-1] * 24 * 3600
	time_steps = 10000
	dt = total_time / time_steps
	t_space = np.linspace(0, total_time, time_steps)

    #Take space_steps = 50 for now, 100 is optimum, it was 30 before: Sept 5/2024
	space_steps = 30
	
	dx = vs.membrane_thickness_m / space_steps
	x_space = np.linspace(0, vs.membrane_thickness_m, space_steps)
	time = np.linspace(vs['time_shift_sec'], total_time, 1000) # seconds

	
	def hot_side_concentration(t):
		# Time-dependent concentration of radon on the hot side of the membrane in Bq/m^3
	
		return vs.C_H * np.exp(-vs.lambda_ * t)
		# return C_H * np.ones_like(t)
	
	
	# DEFINE DIFFERENTIAL EQUATIONS
	def pde_InMembrane(t, C):
	
		dCdt = np.zeros(space_steps)
		
		# Dirichlet boundary condition at x = 0
		C[0] = hot_side_concentration(t)
		
		# dC/dt = D d^2C /dx^2 - lambda * C
		for i in range(1, space_steps-1):
			dCdt[i] = D * (C[i + 1] - 2 * C[i] + C[i - 1]) / dx**2  - vs.lambda_ * C[i]
	
		# Neumann boundary condition at x = H
		dCdt[-1] = - D * (C[-1] - C[-3]) / (2*dx) - vs.lambda_ * C[-1]

		# print(f'> D = {D:0.10e} m^2/s, S = {S:0.10f}, tau_cold = {tau_cold:0.10f}')
	
		return dCdt
	
	
	def pde_coldSide(t, C):
	
		# dC/dt = (A/V) * J - lambda * C
		#	 J = -S * D * dC/dx ~ -S * D * (C(x + dx) - C(x - dx)) / (2*dx)
		dCdt = - P * (vs.membrane_area_m2/vs.V_coldSide_m3) * (solm.sol(t)[-1] - solm.sol(t)[-3]) / (2*dx) - vs.lambda_rad7 * C

		# print(f'>> D = {D:0.10e} m^2/s, S = {S:0.10f}, tau_cold = {tau_cold:0.10f}')
		
		return dCdt
	

	def model_2P(t, par_D, par_P):
	
		global solm
		global D, P
		D, P = par_D, par_P

		# print(f'D = {D:0.10e} m^2/s, S = {S:0.10f}, tau_cold = {tau_cold:0.10f}')

		# D = 10**(D_exp)
		# vs.lambda_rad7 = 1 / tau_cold / 86400 # 1/s
	
		# calculate radon concentration within membrane
		C0 = np.zeros(space_steps)
		solm = solve_ivp(pde_InMembrane,t_span = [t_space[0], t_space[-1]], y0=C0, t_eval=t_space, method='RK45', dense_output=True, )
	
		# print('Finished solving for radon concentration within the membrane.', D, vs.C_H)
	
		# calculate cold-side radon concentration
		C0 = [vs.C_initial]
		sol = solve_ivp(pde_coldSide, t_span = [t_space[0], t_space[-1]], y0=C0, t_eval=t_space, method='RK45', dense_output=True)
	
		# print('Finished solving for radon concentration in the cold-side volume.', D, vs.C_H)

		if verbose==True:
			print(f'D = {D:0.5e} m^2/s, P = {P:0.5e} m^2/s, tau_cold = {vs.tau_cold:0.5f}')
		
		C = sol.sol(t)[0]
	
		return C
	
	
	def model_3P(t, par_D, par_P, par_tau_cold):
	
		global solm
		global D, P, tau_cold
		D, P, tau_cold = par_D, par_P, par_tau_cold

		# print(f'D = {D:0.10e} m^2/s, S = {S:0.10f}, tau_cold = {tau_cold:0.10f}')

		# D = 10**(D_exp)
		vs.lambda_rad7 = 1 / tau_cold / 86400 # 1/s
	
		# calculate radon concentration within membrane
		C0 = np.zeros(space_steps)
		solm = solve_ivp(pde_InMembrane,t_span = [t_space[0], t_space[-1]], y0=C0, t_eval=t_space, method='RK45', dense_output=True, )
	
		# print('Finished solving for radon concentration within the membrane.', D, vs.C_H)
	
		# calculate cold-side radon concentration
		C0 = [vs.C_initial]
		sol = solve_ivp(pde_coldSide, t_span = [t_space[0], t_space[-1]], y0=C0, t_eval=t_space, method='RK45', dense_output=True)
	
		# print('Finished solving for radon concentration in the cold-side volume.', D, vs.C_H)

		if verbose==True:
			print(f'D = {D:0.5e} m^2/s, P = {P:0.5e} m^2/s, tau_cold = {tau_cold:0.5f} days.')
		
		C = sol.sol(t)[0]
	
		return C


	# define data variables (for convenience)
	X = vs.dfc.Seconds
	Y = vs.dfc.RnConc
	U = vs.dfc.Uncert_RnConc

	# print('Y', Y)

	if minimize_model==True:
		
		# Guess and Bounds
		if p0 is None:
			p0 = (1e-13, 1e-13, 2)
			
		if bounds is None:
			bounds = ([1e-15, 1e-15, 0.1], [7e-13, 1e-11 ,5.51])

		print('p0', p0)

		if len(p0) == 2:

			bounds = (bounds[0][:-1], bounds[1][:-1])
			model = model_2P

		if len(p0) == 3:

			model = model_3P

		

		popt, pcov = curve_fit(model, xdata=X, ydata=Y, sigma=U, p0=p0, method='lm', epsfcn=0.1, xtol=1e-14, ftol=1e-10)

		# DEPRICATED
		# popt, pcov = curve_fit(model, xdata=X, ydata=Y, sigma=U, p0=p0, method='trf', # method='lm', epsfcn=0.1, ftol=1e-10)

		# USING: least_squares
		# popt, pcov = curve_fit(model, xdata=X, ydata=Y, sigma=U, p0=p0, bounds=bounds, method='trf', loss='soft_l1', diff_step=[0.2, 0.2, 0.2], xtol=1e-16, ftol=1e-15, gtol=1e-15)
		
		# popt, pcov = least_squares(model, xdata=X, ydata=Y, sigma=U, p0=p0, method='lm', epsfcn=0.1, xtol=1e-14, ftol=1e-10)
		# popt, pcov = leastsq(model, xdata=X, ydata=Y, sigma=U, p0=p0, method='lm', epsfcn=0.1, xtol=1e-14, ftol=1e-10)
		
		uncerts = np.sqrt(np.diag(pcov))

		C_expected = model(X, *popt)

		# Best-fit Parameters and Goodness of Fit
		gof = goodnessOfFit(Y, C_expected, U, len(popt))
		if verbose==True:
			print(f'\npopt = {popt} +/- {uncerts}\npcov = {pcov}')
			print(f'chi squared / ndf = {gof[0]:.2f} / {gof[1]:.0f} = {gof[0] / gof[1]:.2f}, p-value = {gof[3]:.3e}')

		# Call model
		C = model(time, *popt)
		# print('Check', C[-1])
		# C = model(time, D, S, tau_cold)

	else:
		# ALWAYS USING 3P model
		model = model_3P

		# Call model
		C = model(time, D_val, P_val, tau_cold_val)
		# print('Check model', D_val, S_val, tau_cold_val)

		time_expected = vs.dfc.Seconds
		time_expected_days = time_expected / 86400
		C_expected = model(time_expected, D, P, tau_cold)
		# print('Check model expected:', D, S, tau_cold)

		# Best-fit Parameters and Goodness of Fit
		gof = goodnessOfFit(Y, C_expected, U, 3)
		if verbose==True:
			print(f'chi squared / ndf = {gof[0]:.2f} / {gof[1]:.0f} = {gof[0] / gof[1]:.2f}, p-value = {gof[3]:.3e}')

	if return_expected==True and returnGOF==True:
		return time_expected_days, C_expected, gof # popt, pcov
		
	if return_expected==True:
		return time_expected_days, C_expected # popt, pcov
		
	else:
		return time, C # popt, pcov

		