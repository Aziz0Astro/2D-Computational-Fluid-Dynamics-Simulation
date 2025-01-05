#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:47:07 2024

@author: abdulazizabdulaziz
"""
# Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
    
# Grid measurements
grid_height = 50
grid_width = 50
rectangle_height = 30
rectangle_width = 30
rect_y_i = grid_height/2 - rectangle_height/2
rect_y_f = grid_height/2 + rectangle_height/2
rect_x_i = grid_width/2 - rectangle_width/2
rect_x_f = grid_width/2 + rectangle_width/2

# Create a grid
Phi = np.zeros((grid_height, grid_width))
# Initial conditions
dx = 1 # m
dy = 1 # m
v0 = 1 # flow velocity m/s
angle = np.deg2rad(135) # Angle between the velocity vector and postive x-axis

# Fluid does not flow through the rectangle
Phi[int(rect_y_i):int(rect_y_f),int(rect_x_i):int(rect_x_f)] = 0

            
# function to compute the residuls of the relaxation methods applied on an array.
# If there is at least one residual that is more than the desired r, the 
# function will return False. Otherwise, it will retun True.
def error_tolerance(array, r):
    residual = np.zeros((grid_height,grid_width)) # array to store residuals
    count = 0 # number of residuals that do not satisfy the desired tolerance
    # Loop through every element in the array
    for i in range(len(array)):
        for j in range(len(array[0])):
            # If it is a boundary condition element
            if i==0 and j==0:
                residual[i][j] = (2*array[i+1][j] - v0*np.sin(angle)*2*dy
                                          + 2*array[i][j+1] - v0*np.cos(angle)*2*dx- 4*array[i][j])/(dx**2)
            elif i==len(Phi)-1 and j==0:
                residual[i][j] = (2*array[i-1][j] + v0*np.sin(angle)*2*dy
                                          + 2*array[i][j+1] - v0*np.cos(angle)*2*dx- 4*array[i][j])/(dx**2)
            elif i==0 and j==len(Phi[0])-1:
                residual[i][j] = (2*array[i+1][j] - v0*np.sin(angle)*2*dy
                                          + 2*array[i][j-1] + v0*np.cos(angle)*2*dx- 4*array[i][j])/(dx**2)
            elif i==len(Phi)-1 and j==len(Phi[0])-1:
                residual[i][j] = (2*array[i-1][j] + v0*np.sin(angle)*2*dy
                                          + 2*array[i][j-1] + v0*np.cos(angle)*2*dx- 4*array[i][j])/(dx**2)

            elif i==0 and j!=0 and j!=len(Phi[0])-1:
                residual[i][j] = (2*array[i+1][j] - v0*np.sin(angle)*2*dy
                                          + array[i][j+1] + array[i][j-1]- 4*array[i][j])/(dx**2)

            elif i==len(Phi)-1 and j!=0 and j!=len(Phi[0])-1:
                residual[i][j] = (2*array[i-1][j] + v0*np.sin(angle)*2*dy
                                          + array[i][j+1] + array[i][j-1]- 4*array[i][j])/(dx**2)

            elif j==0 and i!=0 and i!=len(Phi)-1:
                residual[i][j] = (2*array[i][j+1] - v0*np.cos(angle)*2*dx
                                          + array[i-1][j] + array[i+1][j]- 4*array[i][j])/(dx**2)
            
            elif j==len(Phi[0])-1 and i!=0 and i!=len(Phi)-1:
                residual[i][j] = (2*array[i][j-1] + v0*np.cos(angle)*2*dx
                                          + array[i-1][j] + array[i+1][j]- 4*array[i][j])/(dx**2)

            elif i == rect_y_f+1 and j >= rect_x_i and j <= rect_x_f:
                residual[i][j] = (array[i+1][j] + array[i+1][j] + 
                                        array[i][j-1] + array[i][j+1]- 4*array[i][j])/(dx**2)

            elif i == rect_y_i-1 and j >= rect_x_i and j <= rect_x_f:
                residual[i][j] = (array[i-1][j] + array[i-1][j] + 
                                        array[i][j-1] + array[i][j+1]- 4*array[i][j])/(dx**2)

            elif i >= rect_y_i and i <= rect_y_f and j == rect_x_i-1:
                residual[i][j] = (array[i+1][j] + array[i-1][j] + 
                                        array[i][j-1] + array[i][j-1]- 4*array[i][j])/(dx**2)

            elif i >= rect_y_i and i <= rect_y_f and j == rect_x_f+1:
                residual[i][j] = (array[i+1][j] + array[i-1][j] + 
                                        array[i][j+1] + array[i][j+1]- 4*array[i][j])/(dx**2)
                
            # If it is not a boundary condition element
            elif i != 0 and j != 0 and i != (len(array)-1) and j != (len(array[0])-1) and not(i >= rect_y_i-1 and i <= rect_y_f+1 and j >= rect_x_i-1 and j <= rect_x_f+1):
                # compute residual
                residual[i][j] = (array[i-1][j] + array[i+1][j] + 
                                  array[i][j-1] + array[i][j+1] - 4*array[i][j])/(dx**2)
    # Check all residuals. If at least one of them is bigger or equal to r, 
    # return False. Otherwise, True.
    for i in range(len(residual)):
        for j in range(len(residual[0])):
            if abs(residual[i][j])>=r:
                count += 1
                
    if count == 0:
        return True
    else:
        return False

Phi_Gauss_Seidel = np.copy(Phi) # Create a copy of Phi

steps_Gauss_Seidel = 0 # Relaxation steps using Gauss_Seidel

# While there is at least one residual not less than 0.0001
while not error_tolerance(Phi_Gauss_Seidel, 0.0001):

    # Perform the Gauss_Seidel method
    for i in range(len(Phi)):

        for j in range(len(Phi[0])):
            # Bottom-left corner
            if i==0 and j==0:
                Phi_Gauss_Seidel[i][j] = (2*Phi_Gauss_Seidel[i+1][j] - v0*np.sin(angle)*2*dy
                                          + 2*Phi_Gauss_Seidel[i][j+1] - v0*np.cos(angle)*2*dx)/4
            # Top-left corner
            elif i==len(Phi)-1 and j==0:
                Phi_Gauss_Seidel[i][j] = (2*Phi_Gauss_Seidel[i-1][j] + v0*np.sin(angle)*2*dy
                                          + 2*Phi_Gauss_Seidel[i][j+1] - v0*np.cos(angle)*2*dx)/4
            # Bottom-right corner
            elif i==0 and j==len(Phi[0])-1:
                Phi_Gauss_Seidel[i][j] = (2*Phi_Gauss_Seidel[i+1][j] - v0*np.sin(angle)*2*dy
                                          + 2*Phi_Gauss_Seidel[i][j-1] + v0*np.cos(angle)*2*dx)/4
            # Top-right corner
            elif i==len(Phi)-1 and j==len(Phi[0])-1:
                Phi_Gauss_Seidel[i][j] = (2*Phi_Gauss_Seidel[i-1][j] + v0*np.sin(angle)*2*dy
                                          + 2*Phi_Gauss_Seidel[i][j-1] + v0*np.cos(angle)*2*dx)/4
            # Bottom wall
            elif i==0 and j!=0 and j!=len(Phi[0])-1:
                Phi_Gauss_Seidel[i][j] = (2*Phi_Gauss_Seidel[i+1][j] - v0*np.sin(angle)*2*dy
                                          + Phi_Gauss_Seidel[i][j+1] + Phi_Gauss_Seidel[i][j-1])/4
            # Top wall
            elif i==len(Phi)-1 and j!=0 and j!=len(Phi[0])-1:
                Phi_Gauss_Seidel[i][j] = (2*Phi_Gauss_Seidel[i-1][j] + v0*np.sin(angle)*2*dy
                                          + Phi_Gauss_Seidel[i][j+1] + Phi_Gauss_Seidel[i][j-1])/4
            # Left wall
            elif j==0 and i!=0 and i!=len(Phi)-1:
                Phi_Gauss_Seidel[i][j] = (2*Phi_Gauss_Seidel[i][j+1] - v0*np.cos(angle)*2*dx
                                          + Phi_Gauss_Seidel[i-1][j] + Phi_Gauss_Seidel[i+1][j])/4
            # Right wall
            elif j==len(Phi[0])-1 and i!=0 and i!=len(Phi)-1:
                Phi_Gauss_Seidel[i][j] = (2*Phi_Gauss_Seidel[i][j-1] + v0*np.cos(angle)*2*dx
                                          + Phi_Gauss_Seidel[i-1][j] + Phi_Gauss_Seidel[i+1][j])/4
            # Right rectangle side
            elif i == rect_y_f+1 and j >= rect_x_i and j <= rect_x_f:
                Phi_Gauss_Seidel[i][j] = (Phi_Gauss_Seidel[i+1][j] + Phi_Gauss_Seidel[i+1][j] + 
                                        Phi_Gauss_Seidel[i][j-1] + Phi_Gauss_Seidel[i][j+1])/4
            # Left rectangle side
            elif i == rect_y_i-1 and j >= rect_x_i and j <= rect_x_f:
                Phi_Gauss_Seidel[i][j] = (Phi_Gauss_Seidel[i-1][j] + Phi_Gauss_Seidel[i-1][j] + 
                                        Phi_Gauss_Seidel[i][j-1] + Phi_Gauss_Seidel[i][j+1])/4
            # Bottom rectangle side
            elif i >= rect_y_i and i <= rect_y_f and j == rect_x_i-1:
                Phi_Gauss_Seidel[i][j] = (Phi_Gauss_Seidel[i+1][j] + Phi_Gauss_Seidel[i-1][j] + 
                                        Phi_Gauss_Seidel[i][j-1] + Phi_Gauss_Seidel[i][j-1])/4
            # Top rectangle side
            elif i >= rect_y_i and i <= rect_y_f and j == rect_x_f+1:
                Phi_Gauss_Seidel[i][j] = (Phi_Gauss_Seidel[i+1][j] + Phi_Gauss_Seidel[i-1][j] + 
                                        Phi_Gauss_Seidel[i][j+1] + Phi_Gauss_Seidel[i][j+1])/4
            # If it is not a boundary condition
            elif i != 0 and i != len(Phi)-1 and j != 0 and j != len(Phi[0])-1 and not(i >= rect_y_i and i <= rect_y_f and j >= rect_x_i and j <= rect_x_f):
                Phi_Gauss_Seidel[i][j] = (Phi_Gauss_Seidel[i-1][j] + Phi_Gauss_Seidel[i+1][j] + 
                                        Phi_Gauss_Seidel[i][j-1] + Phi_Gauss_Seidel[i][j+1])/4

                

                
                
                
              
    steps_Gauss_Seidel += 1
    print(steps_Gauss_Seidel)
    
# Calculate the gradient of the potential function or the velocity components
u, v = np.gradient(Phi_Gauss_Seidel)
# Clear the noise from the edges of the rectangle
u[int(rect_y_i),int(rect_x_i):int(rect_x_f+1)] = 0
v[int(rect_y_i),int(rect_x_i):int(rect_x_f+1)] = 0
u[int(rect_y_f),int(rect_x_i):int(rect_x_f+1)] = 0
v[int(rect_y_f),int(rect_x_i):int(rect_x_f+1)] = 0
u[int(rect_y_i):int(rect_y_f+1),int(rect_x_i)] = 0
v[int(rect_y_i):int(rect_y_f+1),int(rect_x_i)] = 0
u[int(rect_y_i):int(rect_y_f+1),int(rect_x_f)] = 0
v[int(rect_y_i):int(rect_y_f+1),int(rect_x_f)] = 0


# Create plot grid
x = np.arange(0, 50)
y = np.arange(0, 50)

X, Y = np.meshgrid(x, y)

# Plot the velocity field
fig, ax = plt.subplots(figsize=(10,10))

skip = (slice(None, None, 2), slice(None, None, 2)) # Control the density of the arrows
ax.quiver(X[skip], Y[skip], v[skip], u[skip], color='black',
           headwidth=5, scale=60, headlength=4)
plt.title("Airflow Velocity Field Around a Rectangle")
# Create a Rectangle patch
rect = patches.Rectangle((rect_x_i, rect_y_i), rectangle_width, rectangle_height, linewidth=1, edgecolor='r', facecolor='black')
# Add the patch to the grid
ax.add_patch(rect)
plt.show()
plt.close()

# Plot flow potential as a color map
plt.figure(figsize=(10,10))
plt.imshow(Phi_Gauss_Seidel)
plt.title("Airflow Potential Around a Rectangle")
# show plot
plt.show()
plt.close()

    

    
    
    
    
    