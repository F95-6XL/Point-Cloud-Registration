% clc
% clear all
qq = "0411"%411
for a = [600:1:600]
%     qq = '0'+string(a);
    m = load('./Matrix/new_data/max_cons_matrix/lod3_0.04_old/max_cons_matrix_0'+qq+'.txt');%fname0_nv.txt');
%     m = sqrt(m);
    figure
    read_matrix(m,0.04);
    matrix_2d_plot(m,0.04);
%     hold on 
%     x = [-2,2];
%     y = [-2,2];
%     z = ones(2)*1552.94*0.0;
%     surf(x,y,z)
%     alpha(0.7)   
    xlim([-2 2])
    ylim([-2 2])
%     
% 
%     matrix_2d_plot_with_80p(m,0.04);
    
%     figure
%     matrix_x(m,0.04)
%     figure
%     read_matrix(m,0.2,1);
%     zlabel('Occurance')
%     title('accumulator of consensus set')
%     xlim([-2 2])
%     ylim([-2 2])

%     figure
%     m = Gauss_filter(m, 10, 201)
%     read_matrix(m)
%     title('Maximum consensus matrix')

    figure
    c = load('./Matrix/new_data/cov_matrix/lod3_0.04/cov_matrix_0'+qq+'.txt');
    c = sqrt(c);
    read_matrix(c,0.04)
    xlim([-2 2])
    ylim([-2 2])
    matrix_2d_plot(c,0.04);



%     figure
%     c = load('./Matrix/new_data/hough_matrix/max_cons_matrix/lod3_0.04/hough_matrix_0'+qq+'.txt');
%     read_hough_matrix(c)
%     figure
%     c = load('./Matrix/new_data/hough_matrix/cov_matrix/lod3_0.04/hough_matrix_0'+qq+'.txt');
%     c = round(c*1000/521);
%     read_hough_matrix(c')

end  

% % c = load('./Matrix/new_data/cov_matrix/lod3_0.04_hard_radius0.06_maxnn50000/cov_matrix_0'+qq+'.txt');
% % c = load('./Matrix/new_data/cov_matrix/lod2_0.04_downsampled_voxel0.2/cov_matrix_0'+qq+'.txt');

% zlabel('Occurance')
% title('accumulator of cov\_score')
% xlim([-2 2])
% ylim([-2 2])

% figure
% c = Gauss_filter(c, 10, 101)
% read_matrix(c)

%%% plot hough space accumulator
% figure
% c = load('./Matrix/new_data/hough_matrix/max_cons_matrix/lod3_0.04/hough_matrix_0'+qq+'.txt');
% read_hough_matrix(c)
% figure
% c = load('./Matrix/new_data/hough_matrix/cov_matrix/lod3_0.04/hough_matrix_0'+qq+'.txt');
% read_hough_matrix(c)


%%% plot top view of matrix
% matrix_2d_plot(c,0.04,1.2, [0,0])



% figure

%read_matrix(zeros(41,41),0.1,2)




function read_matrix(datamatrix,grid_size,search_range)
[xsize, ysize] = size(datamatrix);
search_range = (xsize-1)*grid_size/2;
xRow = -search_range:grid_size:search_range;%(0:xsize-1)%
yCol = -search_range:grid_size:search_range;%(0:ysize-1)%
x=repmat(xRow',1,ysize);
y=repmat(yCol,xsize,1);

[X,Y]=meshgrid(xRow,yCol);
Z=griddata(x,y,datamatrix,X,Y,'cubic');%插值，重构用于画图的Z轴数据

clear a b xmax xmin ymin ymax xRow yCol z;
mesh(X,Y,Z);
title("Accumulator of consensus sets");
xlabel("x in [m]");
ylabel("y in [m]");
zlabel("consensus score")
view([-15,15]);

hold on
s = surf(X,Y,Z);

% x = [-2,2];
% y = [-2,2];
% z = 800*ones(2);
% a =surf(x,y,z)
% alpha(a,0.8)   
end

function read_hough_matrix(datamatrix)
[xsize, ysize] = size(datamatrix);
xRow = (0:xsize-1)%
yCol = (0:ysize-1)%
x=repmat(xRow',1,ysize);
y=repmat(yCol,xsize,1);

[X,Y]=meshgrid(xRow,yCol);
Z=griddata(x,y,datamatrix,X,Y,'cubic');%插值，重构用于画图的Z轴数据

clear a b xmax xmin ymin ymax xRow yCol z;
mesh(X,Y,Z);
title('Hough space accumulator');
ylabel('rho in [m]');
xlabel('theta in [deg]');
xlim([0 180]);
hold on
s = surf(X,Y,Z);


figure
[xsize, ysize] = size(datamatrix)

x = (0:xsize-1);
y = (0:ysize-1);
[X, Y] = meshgrid(x, y)
Z = [datamatrix'];
contourf(X, Y, Z);
pcolor(X, Y, Z);
shading interp
colorbar
title('accumulator of consensus sets')
xlabel('x in [m]')
ylabel('y in [m]')
end

function gauss=Gauss_filter(Matrix, sigma, window)
window = double(uint8(window));
Mask=fspecial('gaussian', window, sigma);
gauss=imfilter(Matrix,Mask)%,'X');
end

function matrix_2d_plot(datamatrix, grid_size)
figure
[xsize, ysize] = size(datamatrix);
search_range = (xsize-1)*grid_size/2;
x = -search_range:grid_size:search_range;
y = -search_range:grid_size:search_range;
[X, Y] = meshgrid(x, y);
Z = [datamatrix'];
contourf(X, Y, Z);
pcolor(X, Y, Z);
shading interp
title('Accumulator of consensus sets')
xlim([-2 2])
ylim([-2 2])
xlabel('x in [m]')
ylabel('y in [m]')
colorbar

end

function matrix_2d_plot_with_blob(datamatrix, grid_size, blob_size, shift)
figure
[xsize, ysize] = size(datamatrix);
search_range = (xsize-1)*grid_size/2;
x = -search_range:grid_size:search_range;
y = -search_range:grid_size:search_range;
[X, Y] = meshgrid(x, y);
Z = [datamatrix'];
contourf(X, Y, Z);
pcolor(X, Y, Z);
shading interp
title('Accumulator of consensus sets')
xlim([-2 2])
ylim([-2 2])
xlabel('x in [m]')
ylabel('y in [m]')
colorbar
hold on

% plot blob
blob_size = blob_size*grid_size;
[x_h, y_h]= find(datamatrix==max(max(datamatrix)));
x_h = grid_size*(x_h -1 - (xsize-1)/2 + shift(1))
y_h = grid_size*(y_h -1 - (xsize-1)/2 + shift(2))

theta=0:0.01:2*pi;
Circle1=x_h+blob_size*cos(theta);
Circle2=y_h+blob_size*sin(theta);
c=[255,255,0]; % color of circle
plot(Circle1,Circle2,'r','linewidth',1);
axis equal

end

function matrix_2d_plot_with_80p(datamatrix, grid_size)
figure
[xsize, ysize] = size(datamatrix);
search_range = (xsize-1)*grid_size/2;
x = -search_range:grid_size:search_range;
y = -search_range:grid_size:search_range;
[X, Y] = meshgrid(x, y);
Z = [datamatrix'];
contourf(X, Y, Z);
pcolor(X, Y, Z);
shading interp
title('Accumulator of consensus sets')
xlabel('x in [m]')
ylabel('y in [m]')
xlim([-2 2])
ylim([-2 2])
colorbar
hold on

% plot error ellipse w.r.t. 80%
% maximum = max(datamatrix,[],'all');
% [x y] = find(datamatrix>maximum*0.8);
% points = ([x y]-1) * grid_size - search_range
% ErrorEllipse(points,0.998689492137086,5)


axis equal

end

function ErrorEllipse(datamatrix,p,marksize)
data = datamatrix;

% 计算协方差矩阵、特征向量、特征值
covariance = cov(data);
[eigenvec, eigenval ] = eig(covariance);

% 求取最大特征向量
[largest_eigenvec_ind_c, r] = find(eigenval == max(max(eigenval)));
largest_eigenvec = eigenvec(:, largest_eigenvec_ind_c);

% 求取最大特征值
largest_eigenval = max(max(eigenval));

% 计算最小特征向量和最小特征值

if(largest_eigenvec_ind_c == 1)
    smallest_eigenval = max(eigenval(:,2));
    smallest_eigenvec = eigenvec(:,2);
else
    smallest_eigenval = max(eigenval(:,1));
    smallest_eigenvec = eigenvec(1,:);
end

% 计算X轴和最大特征向量直接的夹角，值域为[-pi,pi]
angle = atan2(largest_eigenvec(2), largest_eigenvec(1));

% 当夹角为负时，加2pi求正值

if(angle < 0)
    angle = angle + 2*pi;
end

% 计算数据的两列均值，格式为2乘1的矩阵
avg = mean(data);

% 配置置信椭圆的参数，包括卡方值、旋转角度、均值、长短轴距
chisquare_val = sqrt(chi2inv(p,2));
theta_grid = linspace(0,2*pi);
phi = angle;
X0=avg(1);
Y0=avg(2);
a=chisquare_val*sqrt(largest_eigenval);
b=chisquare_val*sqrt(smallest_eigenval);

% 将椭圆投射到直角坐标轴中 
ellipse_x_r  = a*cos( theta_grid );
ellipse_y_r  = b*sin( theta_grid );

R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];% 旋转矩阵
r_ellipse = [ellipse_x_r;ellipse_y_r]' * R;% 相乘，旋转椭圆

plot(r_ellipse(:,1) + X0,r_ellipse(:,2) + Y0,'r','linewidth',1)% 打印置信椭圆
hold on;

% plot(data(:,1), data(:,2), '.','markersize',marksize);% 打印原始数据


hold on;
end

function matrix_x(datamatrix,grid_size)
[xsize, ysize] = size(datamatrix);
search_range = (xsize-1)*grid_size/2;
xRow = -search_range:grid_size:search_range;
a = max(datamatrix);
plot(xRow, a)

end

