import numpy as np
import matplotlib.pyplot as plt

#source: https://github.com/JohnBracken/Bland-Altman-and-correlation-analysis-in-Python/blob/master/BlandAltman_correlation.py

#Function to create a Bland-Altman plot comparing agreement between two data sets.
def BlandAltmanPlot(Xdata, Ydata):


    #Calculate the difference between the CACT and actual angles.  Will create a difference array comparing each angle
    Difference = Ydata - Xdata


    #Calculate the average value of the two datasets for each data point
    Average = (Xdata + Ydata)/2


    #Calculate the mean of the differences.
    Mean_difference = np.mean(Difference)


    #Calculate the sample standard deviation of the difference.
    Std_difference = np.std(Difference)


    #Calculate the upper and lower limits of the agreement (95% confidence).
    upper_limit = Mean_difference + 1.96*Std_difference
    lower_limit = Mean_difference - 1.96*Std_difference


    #Set axis limits, this will account for the maximum average difference and the upper and lower
    #limits of agreement, and all data points.
    plt.axis([0, np.max(Average)+5, np.min(Difference)-5, np.max(Difference)+5],
             fontsize = '16', lw = '2')


    #Do the Bland-Altman plot.
    plt.plot(Average, Difference, marker ='o', mfc = 'none', ls = 'None', mec = 'blue',
             mew= '2', ms = '8', lw = '2')


    #Add the mean, upper and lower levels of agreement to the Bland-Altman plot.
    plt.axhline(y=Mean_difference, lw ='2', color ='k', ls = 'dashed')
    plt.axhline(y=upper_limit,lw ='2', color ='k', ls = 'dashed')
    plt.axhline(y=lower_limit,lw ='2', color ='k', ls = 'dashed')


    #Horizontal axis label
    plt.xlabel('Average difference', fontsize = '16')


    #Vertical axis label
    plt.ylabel('Difference', fontsize = '16')


    #Change the font size of the tick mark labels to be bigger
    #on both axes.
    plt.tick_params(axis='both', which='major', labelsize= '16', width = '2')


    #Another way to set the border thickness.  Get current axes first
    #Then change their thickness using the set_linewidth() command.
    ax1 =plt.gca()
    ax1.spines['top'].set_linewidth('2')
    ax1.spines['left'].set_linewidth('2')
    ax1.spines['right'].set_linewidth('2')
    ax1.spines['bottom'].set_linewidth('2')

    plt.show()
    #End the function
    return;