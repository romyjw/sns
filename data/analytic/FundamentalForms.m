syms r theta phi scale real

% Define the parametric equations for a sphere in spherical coordinates
%r = [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)];


%SPIKE
%r = scale * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * (1+0.4*sin(theta)^2 * sin(6*theta) * sin(6*phi))


%LAUNDRY
%P *  np.transpose(( np.transpose(np.ones_like(P))  +   0.1 * (0.4*((U*(np.pi-U))**2)*np.cos(12*U)**2 + 0.4*((U*(np.pi-U))**2)*np.sin(12*V +     6*U)**2) ))
r = scale * [sin(theta)*cos(phi); sin(theta)*sin(phi); cos(theta)] * ( 1 + 0.1 * (0.4*((theta*(pi-theta))^2)*cos(12*theta)^2 + 0.4*((theta*(pi-theta))^2)*sin(12*phi +     6*theta)^2) )



r_theta = simplify(diff(r, theta))
r_phi = simplify(diff(r, phi))

% Calculate the first fundamental form coefficients
E = simplify(dot(r_theta, r_theta));
F = simplify(dot(r_theta, r_phi));
G = simplify(dot(r_phi, r_phi));

% Calculate the normals
n = simplify(cross(r_theta,r_phi) / norm(cross(r_theta,r_phi)));
disp('n1');
disp(n(1))
disp('n2');
disp(n(2))
disp('n3');
disp(n(3))

r_theta_theta = simplify(diff(r_theta, theta));
r_theta_phi = simplify(diff(r_theta, phi));
r_phi_phi = simplify(diff(r_phi, phi));

% Calculate the second fundamental form coefficients
L = simplify(dot(n, r_theta_theta));
M = simplify(dot(n, r_theta_phi));
N = simplify(dot(n, r_phi_phi));

% Calculate the mean curvature
H = simplify((E*N - 2*F*M + G*L) / (2*(E*G - F^2)));
disp('Mean Curvature:');
disp(H)

K = simplify((L*N - M^2) / (E*G - F^2));
disp('Gauss Curvature:');
disp(K)

%coefficients of quadratic eigenvalue equation
A = simplify ( E*G-F^2);
B = simplify( 2*M*F - (E*N+L*G));
C = simplify (L*N - M^2);

disc =  (B^2 - 4*A*C);

k1 =  ((-B - sqrt(disc))/(2*A));
k2 = ((-B + sqrt(disc))/(2*A));

e1 = r_theta;%simplify( r_theta / sqrt(dot(r_theta,r_theta)))
e2 = r_phi;%simplify( r_phi / sqrt(dot(r_phi,r_phi)))

x1 =  ( (k1*E - L)/(M-k1*F) );
x2 = ( (k2*E-L)/(M-k2*F) );

dir1 = ( e1 + x1*e2 );

disp('dir11');
disp(dir1(1))
disp('dir12');
disp(dir1(2))
disp('dir13');
disp(dir1(3))

dir2 = ( e1 + x2*e2 );
disp('dir21');
disp(dir2(1))
disp('dir22');
disp(dir2(2))
disp('dir23');
disp(dir2(3))



%ndir1 = simplify(dir1/sqrt(dot(dir1,dir1)))
%ndir2 = simplify(dir2/sqrt(dot(dir2,dir2)))

%ortho = simplify(dot(ndir1,ndir2))


% Display the mean curvature

%theta=0.5;
%phi=0.2;
%disp('ortho');
%disp(subs(ortho))

