from cmath import nan
import numpy as np
# Z calculado
psi_d = 0 *(np.pi/180)
theta_d = -90 *(np.pi/180)
phi_d = 0 *(np.pi/180)
# Rot Matrix desired
R_d = np.zeros((3,3))
R_d[0,0] = np.cos(theta_d)*np.cos(phi_d); R_d[0,1] = np.sin(psi_d)*np.sin(theta_d)*np.cos(phi_d) - np.cos(psi_d)*np.sin(phi_d);R_d[0,2] = np.cos(psi_d)*np.sin(theta_d)*np.cos(phi_d) + np.sin(psi_d)*np.sin(phi_d)
R_d[1,0] = np.cos(theta_d)*np.sin(phi_d); R_d[1,1] = np.sin(psi_d)*np.sin(theta_d)*np.sin(phi_d) + np.cos(psi_d)*np.cos(phi_d);R_d[1,2] = np.cos(psi_d)*np.sin(theta_d)*np.sin(phi_d) - np.sin(psi_d)*np.cos(phi_d)
R_d[2,0] = -np.sin(theta_d); R_d[2,1] = np.sin(psi_d)*np.cos(theta_d); R_d[2,2] = np.cos(psi_d)*np.cos(theta_d)

print('R_d');print(R_d);print()

Z = [-1.00,	0.00,	0.00]
#Z = [0,	 0.9999999,0]
Z_squared = [z ** 2 for z in Z]
Z_normalized = Z / np.sqrt(np.sum(Z_squared))
#print(Z_normalized)

if  all(v == 0 for v in np.cross([0,1,0],Z_normalized))== True:   
    X = np.cross(Z_normalized,[1,0,0])
    #print(X)
    print('Different Route')
else:
    X = np.cross([0,1,0],Z_normalized)
    #print(X)

X_squared = [x ** 2 for x in X]
X_normalized = X / np.sqrt(np.sum(X_squared))
#print(X_normalized)

Y = np.cross(Z_normalized,X_normalized)
Y_squared = [y ** 2 for y in Y]
Y_normalized = Y / np.sqrt(np.sum(Y_squared)) 

# Rot Matrix in Camera Ref Frame
R_ori = np.zeros((3,3))
# R =[X | Y | Z]
# É este
R_ori[:,0] = X_normalized; R_ori[:,1] = Y_normalized; R_ori[:,2] = Z_normalized
# R = [ X
#       Y
#       Z]
#R_ori[0,:] = X_normalized; R_ori[1,:] = Y_normalized; R_ori[2,:] = Z_normalized
print('R_ori')
print(R_ori)

########################################################################
# Extract Euler angles from Rot Matrix
# Rotate first in x, then in y, and then in z

# psi Ψ - x
# Theta θ- y
# phi φ - z
if R_ori[2,0] != 1:
    print('Veio por aqui 1')
    theta1 = -np.arcsin(R_ori[2,0])
    theta2 = np.pi - theta1
    psi1 = np.arctan2(R_ori[2,1]/np.cos(theta1),R_ori[2,2]/np.cos(theta1))
    psi2 = np.arctan2(R_ori[2,1]/np.cos(theta2),R_ori[2,2]/np.cos(theta2))
    phi1 = np.arctan2(R_ori[1,0]/np.cos(theta1),R_ori[0,0]/np.cos(theta1))
    phi2 = np.arctan2(R_ori[1,0]/np.cos(theta2),R_ori[0,0]/np.cos(theta2))

    print("Ψ = " +str(psi1*180/np.pi) + 'º OR ' +str(psi2*180/np.pi))
    print("θ = " +str(theta1*180/np.pi) + 'º OR ' +str(theta2*180/np.pi))
    print("φ = " +str(phi1*180/np.pi) + 'º OR ' +str(phi2*180/np.pi))
elif  R_ori[2,0] != -1:
    print('Veio por aqui 2')
    theta1 = -np.arcsin(R_ori[2,0])
    theta2 = np.pi - theta1
    psi1 = np.arctan2(R_ori[2,1]/np.cos(theta1),R_ori[2,2]/np.cos(theta1))
    psi2 = np.arctan2(R_ori[2,1]/np.cos(theta2),R_ori[2,2]/np.cos(theta2))
    phi1 = np.arctan2(R_ori[1,0]/np.cos(theta1),R_ori[0,0]/np.cos(theta1))
    phi2 = np.arctan2(R_ori[1,0]/np.cos(theta2),R_ori[0,0]/np.cos(theta2))

    print("Ψ = " +str(psi1*180/np.pi) + 'º OR ' +str(psi2*180/np.pi))
    print("θ = " +str(theta1*180/np.pi) + 'º OR ' +str(theta2*180/np.pi))
    print("φ = " +str(phi1*180/np.pi) + 'º OR ' +str(phi2*180/np.pi))
else:
    phi = 0
    if R_ori[2,0] == -1:
        theta = np.pi/2
        psi = phi + np.arctan2(R_ori[0,1],R_ori[0,2])
    else:
        theta = -np.pi/2
        psi = -phi + np.arctan2(-R_ori[0,1],-R_ori[0,2])

# if R_ori[0,0] == 0 or R_ori[1,0] == 0:
#     phi = 0;
#     theta1 = np.pi/2
#     theta2 = -np.pi/2
#     psi1 = np.arctan2(R_ori[0,1],R_ori[1,1])
#     psi2 = np.arctan2(-R_ori[0,1],-R_ori[1,1])
#     print("Ψx = " +str(psi1*180/np.pi) + 'º OR ' +str(psi2*180/np.pi))
#     print("θy = " +str(theta1*180/np.pi) + 'º OR ' +str(theta2*180/np.pi))
#     print("φz = " +str(phi*180/np.pi) )
  
# else:
#     theta1 = np.arcsin(R_ori[2,0])
#     theta2 = np.pi - theta1
#     psi1 = np.arctan2(R_ori[2,1]/np.cos(theta1),R_ori[2,2]/np.cos(theta1))
#     psi2 = np.arctan2(R_ori[2,1]/np.cos(theta2),R_ori[2,2]/np.cos(theta2))
#     phi1 = np.arctan2(R_ori[1,0]/np.cos(theta1),R_ori[0,0]/np.cos(theta1))
#     phi2 = np.arctan2(R_ori[1,0]/np.cos(theta2),R_ori[0,0]/np.cos(theta2))

#     print("Ψx = " +str(psi1*180/np.pi) + 'º OR ' +str(psi2*180/np.pi))
#     print("θy = " +str(theta1*180/np.pi) + 'º OR ' +str(theta2*180/np.pi))
#     print("φz = " +str(phi1*180/np.pi) + 'º OR ' +str(phi2*180/np.pi))
