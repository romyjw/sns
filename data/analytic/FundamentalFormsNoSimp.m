syms r theta phi scale real

%SPHERE
%r = [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)];

%SPIKE
%fileID = fopen('SPIKE/analytic_result.txt', 'w');
%r = scale * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * (1+0.4*sin(theta)^2 * sin(6*theta) * sin(6*phi))

%LAUNDRY
%fileID = fopen('LAUNDRY/analytic_result.txt', 'w');
%P *  np.transpose(( np.transpose(np.ones_like(P))  +   0.1 * (0.4*((U*(np.pi-U))**2)*np.cos(12*U)**2 + 0.4*((U*(np.pi-U))**2)*np.sin(12*V +     6*U)**2) ))
%r = scale * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * ( 1 + 0.1 * (0.4*((theta*(pi-theta))^2)*cos(12*theta)^2 + 0.4*((theta*(pi-theta))^2)*sin(12*phi +     6*theta)^2) )

%TREE
%fileID = fopen('TREE/analytic_result.txt', 'w');
%P *  np.transpose(( np.transpose(np.ones_like(P))  +   0.05 * (2*((U)**3)*np.sin(6*U)**2 + 0.6*((U*(np.pi-U))**2)*np.sin(6*V +     6*(U - 0.0)**2)**2) ) )
%r = scale * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * (1 + 0.05 * (2*((theta)^3)*sin(6*theta)^2 + 0.6*((theta*(pi-theta))^2)*sin(6*phi +     6*(theta - 0.0)^2)^2) ) 


%SMALLTREE
%fileID = fopen('SMALLTREE/analytic_result.txt', 'w');
% 0.301 * (P *  np.transpose(( np.transpose(np.ones_like(P))  +   0.05 *
% (2*((U)**3)*np.sin(6*U)**2 + 0.6*((U*(np.pi-U))**2)*np.sin(6*V +     6*(U - 0.0)**2)**2) ) ) )
%r = 0.301 * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * (1 + 0.05 * (2*((theta)^3)*sin(6*theta)^2 + 0.6*((theta*(pi-theta))^2)*sin(6*phi +     6*(theta - 0.0)^2)^2) ) 


%SMALLFLOWER
%fileID = fopen('SMALLFLOWER/analytic_result.txt', 'w');
%P *  np.transpose(( np.transpose(np.ones_like(P))  +   0.3 * (0.4*((U*(np.pi-U))**2)*np.cos(18*U)**2 + 0.6*((U*(np.pi-U))**2)*np.sin(6*V +     18*U)**2) ))
%r = 0.366 * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * (1+ 0.3 * (0.4*((theta*(pi-theta))^2)*cos(18*theta)^2 + 0.6*((theta*(pi-theta))^2)*sin(6*phi +     18*theta)^2) )

%SMALLBOBBLE
%fileID = fopen('SMALLBOBBLE/analytic_result.txt', 'w');
%P *  np.transpose(( np.transpose(np.ones_like(P))  +   0.1 * (0.4*((U*(np.pi-U))**2)*np.cos(12*U)**2 + 0.4*((U*(np.pi-U))**2)*np.sin(12*V +     6*U)**2) ))
%r = 0.679 * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * ( 1 + 0.1 * (0.4*((theta*(pi-theta))^2)*cos(12*theta)^2 + 0.4*((theta*(pi-theta))^2)*sin(12*phi +     6*theta)^2) )

%SMALLSPIKE
fileID = fopen('SMALLSPIKE/analytic_result.txt', 'w');
r = 0.772 * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * (1+0.4*sin(theta)^2 * sin(6*theta) * sin(6*phi))


%FLOWER
%fileID = fopen('FLOWER/analytic_result.txt', 'w');
%P *  np.transpose(( np.transpose(np.ones_like(P))  +   0.3 * (0.4*((U*(np.pi-U))**2)*np.cos(18*U)**2 + 0.6*((U*(np.pi-U))**2)*np.sin(6*V +     18*U)**2) ))
%r = scale * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * (1+ 0.3 * (0.4*((theta*(pi-theta))^2)*cos(18*theta)^2 + 0.6*((theta*(pi-theta))^2)*sin(6*phi +     18*theta)^2) )





r_theta = simplify(diff(r, theta))
r_phi = simplify(diff(r, phi))

% Calculate the first fundamental form coefficients
E = dot(r_theta, r_theta);
F = dot(r_theta, r_phi);
G = dot(r_phi, r_phi);

% Calculate the normals
n = cross(r_theta,r_phi) / norm(cross(r_theta,r_phi));

r_theta_theta = diff(r_theta, theta);
r_theta_phi = diff(r_theta, phi);
r_phi_phi = diff(r_phi, phi);

% Calculate the second fundamental form coefficients
L = dot(n, r_theta_theta);
M = dot(n, r_theta_phi);
N = dot(n, r_phi_phi);

% Calculate the mean curvature
H = (E*N - 2*F*M + G*L) / (2*(E*G - F^2));

K = (L*N - M^2) / (E*G - F^2);


%coefficients of quadratic eigenvalue equation
A =  E*G-F^2;
B =  2*M*F - (E*N+L*G);
C = L*N - M^2;

disc =  (B^2 - 4*A*C);

k1 =  ((-B - sqrt(disc))/(2*A));
k2 = ((-B + sqrt(disc))/(2*A));

e1 = r_theta;%simplify( r_theta / sqrt(dot(r_theta,r_theta)))
e2 = r_phi;%simplify( r_phi / sqrt(dot(r_phi,r_phi)))

x1 =  ( (k1*E - L)/(M-k1*F) );
x2 = ( (k2*E-L)/(M-k2*F) );

dir1 = ( e1 + x1*e2 );
dir2 = ( e1 + x2*e2 );






% Convert symbolic expressions to character strings

E_str = char(E);
F_str = char(F);
G_str = char(G);


L_str = char(L);
M_str = char(M);
N_str = char(N);



H_str = char(H);
K_str = char(K);



n1_str = char(n(1));
n2_str = char(n(2));
n3_str = char(n(3));

dir11_str = char(dir1(1));
dir12_str = char(dir1(2));
dir13_str = char(dir1(3));

dir21_str = char(dir2(1));
dir22_str = char(dir2(2));
dir23_str = char(dir2(3));

% Write the character strings to the text file
fprintf(fileID, 'E = %s;', E_str);
fprintf(fileID, 'F = %s;', F_str);
fprintf(fileID, 'G = %s;', G_str);


fprintf(fileID, 'L = %s;', L_str);
fprintf(fileID, 'M = %s;', M_str);
fprintf(fileID, 'N = %s;', N_str);


fprintf(fileID, 'H = %s;', H_str);
fprintf(fileID, 'K = %s;', K_str);

fprintf(fileID, 'n1 = %s;', n1_str);
fprintf(fileID, 'n2 = %s;', n2_str);
fprintf(fileID, 'n3 = %s;', n3_str);

fprintf(fileID, 'dir11 = %s;', dir11_str);
fprintf(fileID, 'dir12 = %s;', dir12_str);
fprintf(fileID, 'dir13 = %s;', dir13_str);

fprintf(fileID, 'dir21 = %s;', dir21_str);
fprintf(fileID, 'dir22 = %s;', dir22_str);
fprintf(fileID, 'dir23 = %s;', dir23_str);


% Close the text file
fclose(fileID);

