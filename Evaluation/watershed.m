m = load('./Matrix/new_data/watershed_matrix/cov_matrix/lod3_0.04/watershed_matrix_01000_1.txt');
m = m*(255/max(m,[],'all'));
m = uint8(255-m);
figure
title('Watershed')
% imshow(m)


a = [1 0];
a = repmat(a,500,1);
b = [-1 0];
b = repmat(b,500,1);
c = [0 1];
c = repmat(c,1,1);
d = [0 -1];
d = repmat(d,1,1);
e = [a; b; c; d];
exx = e(:,1)'*e(:,1);
exy = e(:,1)'*e(:,2);
eyy = e(:,2)'*e(:,2);
Cov = [exx exy;
       exy eyy];
[eigenvec, eigenval ] = eig(inv(Cov))
t=[0:0.01:2*pi];

x=sqrt(eigenval(1,1))*cos(t);
y=sqrt(eigenval(2,2))*sin(t);


% eigv1 = 0.07
% eigv2 = 0.244
% x=sqrt(eigv2)*cos(t);
% y=sqrt(eigv1)*sin(t);

plot(x,y)
xlim([-1 1]);
ylim([-1 1]);
% hold on
% 
% grid minor
% title('Error Ellipse')

% 1/trace(eigenval)