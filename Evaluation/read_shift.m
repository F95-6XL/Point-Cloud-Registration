m = load('.\Results\shift\new_data_02_19\max_cons_matrix\shift_lod3_0.04_hard_radius0.04_maxnn10000.txt');
m2 = load('.\Results\shift\new_data_02_19\cov_matrix\cov_shift_lod3_0.04_hard_radius0.04_maxnn10000.txt');
%m2 = load('.\Results\shift\new_data_02_19\max_cons_matrix\shift_lod3_0.04.txt');
%m2 = load('.\Results\shift\new_data_02_19\max_cons_matrix\shift_lod3_0.04_hard_radius0.02_maxnn1000_scanr0.04.txt');

m11 = m(:,1:2);
% state = m11(:,1) | m11(:,2);
% m11 = m(state,1:2);
m22 = m2(:,1:2);
x = m(:,1);
y = m(:,2);
mean1 = mean(m11)
cov1 = cov(m11);
eig1 = sqrt(eig(cov1))
skewness1 = skewness(m11);
kurtosis1 = kurtosis(m11) -3;


mean2 = mean(m22)
cov2 = cov(m22);
eig2 = sqrt(eig(cov2))
skewness2 = skewness(m22);
kurtosis2 = kurtosis(m22) -3;

figure
plotHisto(m,1,0.04)
xlim([-1 1])
ylim([-1 1])
zlim([0 475])
title('Accumulator of the highest consensus sets')
zlabel('Occurence')
% 
% figure
% plotHisto_1dx(m,1,0.04)
% xlim([-1 1])
% ylim([-1 1])
% zlim([0 1000])
% title('Accumulator of the highest consensus sets')
% zlabel('Occurence')
% figure
% plotHisto_1dy(m,1,0.04)
% xlim([-1 1])
% ylim([-1 1])
% zlim([0 1000])
% title('Accumulator of the highest consensus sets')
% zlabel('Occurence')

figure
plotHisto(m2,1,0.04)
xlim([-1 1])
ylim([-1 1])
zlim([0 550])
title('Accumulator of the highest consensus sets')
zlabel('Occurence')




%draw error ellipse
% 0.6826 for 1-sigma, 0.955 for 2-sigma, 0.997 for 3-sigma
figure
ErrorEllipse(m(:,1:2),0.682689492137086,10)
xlim([-1 1])
ylim([-1 1])
title('Accumulator of the highest consensus sets')

% Errorellipse3d(m2(:,1:3),1-0.682689,'norm')
% xlim([-0.3 0.3])
% ylim([-0.3 0.3])
% zlabel('heading in [deg]')
% figure
% plotHisto_1dx(m2,1,0.04)
% xlim([-1 1])
% ylim([-1 1])
% zlim([0 1000])
% title('Accumulator of the highest consensus sets')
% zlabel('Occurence')
% figure
% plotHisto_1dy(m2,1,0.04)
% xlim([-1 1])
% ylim([-1 1])
% zlim([0 1000])
% title('Accumulator of the highest consensus sets')
% zlabel('Occurence')


figure
ErrorEllipse(m2(:,1:2),0.682689492137086,3)
xlim([-1 1])
ylim([-1 1])
title('Results of the combined ICP approach')


%hosto subplot
% figure
% title('Accumulator of the highest consensus sets')
% subplot(1,2,1)
% plotHisto_1dx(m,1,0.04)
% xlim([-1 1])
% ylim([-1 1])
% zlim([0 1000])
% zlabel('Occurence')
% subplot(1,2,2)
% plotHisto_1dy(m,1,0.04)
% xlim([-1 1])
% ylim([-1 1])
% zlim([0 1000])
% zlabel('Occurence')
% sgtitle('Accumulator of the highest consensus sets','fontsize',11)
% 
% figure
% title('Accumulator of the highest consensus sets')
% subplot(1,2,1)
% plotHisto_1dx(m2,1,0.04)
% xlim([-1 1])
% ylim([-1 1])
% zlim([0 1000])
% zlabel('Occurence')
% subplot(1,2,2)
% plotHisto_1dy(m2,1,0.04)
% xlim([-1 1])
% ylim([-1 1])
% zlim([0 1000])
% zlabel('Occurence')
% sgtitle('Accumulator of the highest consensus sets','fontsize',11)

% Errorellipse3d(m2(:,1:3),1-0.682689,'norm')
% xlim([-0.3 0.3])
% ylim([-0.3 0.3])
% zlabel('heading')
% plotNormal(m)



% n = load('.\Results\shift\LOD3.txt');
% plot(n(:,3))
% xlabel('points')
% ylabel('max-cons-value')



% plot histogram of max cons matrix
function plotHisto(datamatrix,search_range, grid_edge)
x = datamatrix(:,1);
y = datamatrix(:,2);
histogram2(x,y,'XBinEdges',-search_range-grid_edge/2:grid_edge:search_range+grid_edge/2,'YBinEdges',-search_range-grid_edge/2:grid_edge:search_range+grid_edge/2)
% scatter(x,y,0.5,'.')
view([90,0]);
xlabel('x in [m]')
ylabel('y in [m]')
end

function plotHisto_1dx(datamatrix,search_range, grid_edge)
x = datamatrix(:,1);
y = zeros(1915,1);
histogram2(x,y,'XBinEdges',-search_range-grid_edge/2:grid_edge:search_range+grid_edge/2,'YBinEdges',-search_range-grid_edge/2:grid_edge:search_range+grid_edge/2)
% scatter(x,y,0.5,'.')
view([0,0]);
xlabel('x in [m]')
ylabel('y in [m]')
end

function plotHisto_1dy(datamatrix,search_range, grid_edge)
x = zeros(1915,1);
y = datamatrix(:,2);
histogram2(x,y,'XBinEdges',-search_range-grid_edge/2:grid_edge:search_range+grid_edge/2,'YBinEdges',-search_range-grid_edge/2:grid_edge:search_range+grid_edge/2)
% scatter(x,y,0.5,'.')
view([90,0]);
xlabel('x in [m]')
ylabel('y in [m]')
end





% plot a standard normal distribution using mean and covariance matrix of
% peaks
function plotNormal(datamatrix)
x = datamatrix(:,1);
y = datamatrix(:,2);
cov1 = cov(x,y);
mu=[mean(x),mean(y)];% 均值向量
Sigma=cov1;% 协方差矩阵
[X,Y]=meshgrid(-2:0.1:2,-2:0.1:2);%在XOY面上，产生网格数据
p=mvnpdf([X(:) Y(:)],mu,Sigma);%求取联合概率密度，相当于Z轴
p=reshape(p,size(X));%将Z值对应到相应的坐标上

figure
set(gcf,'Position',get(gcf,'Position').*[1 1 1.3 1])

subplot(2,3,[1 2 4 5])
surf(X,Y,p),axis tight,title('2-d standard normal distribution')
subplot(2,3,3)
surf(X,Y,p),view(2),axis tight,title('projection on xy-plane')
subplot(2,3,6)
surf(X,Y,p),view([0 0]),axis tight,title('projection on xz-plane')
end

% plot data and draw confidence ellipse 
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

plot(r_ellipse(:,1) + X0,r_ellipse(:,2) + Y0,'-','linewidth',1)% 打印置信椭圆
hold on;

plot(data(:,1), data(:,2), '.','markersize',marksize);% 打印原始数据
title('Error Ellipse')
xlabel('x in [m]')
ylabel('y in [m]')

hold on;
end

% draw confidence ellipse for 3d data
function Errorellipse3d(xdat,alpha,distribution)
%   绘制置信区域（区间、椭圆区域或椭球区域）
%   ConfidenceRegion(xdat,alpha,distribution)
%   xdat：样本观测值矩阵,p*N 或 N*p的矩阵，p = 1,2 或 3
%   alpha：显著性水平，标量，取值介于[0,1]，默认值为0.05
%   distribution：字符串（'norm'或'experience'），用来指明求置信区域用到的分布类型，
%   distribution的取值只能为字符串'norm'或'experience'，分别对应正态分布和经验分布
%   CopyRight：xiezhh（谢中华）
%   date：2011.4.14
%
%   example1：x = normrnd(10,4,100,1);
%             ConfidenceRegion(x)
%             ConfidenceRegion(x,'exp')
%   example2：x = mvnrnd([1;2],[1 4;4 25],100);
%             ConfidenceRegion(x)
%             ConfidenceRegion(x,'exp')
%   example3：x = mvnrnd([1;2;3],[3 0 0;0 5 -1;0 -1 1],100);
%             ConfidenceRegion(x)
%             ConfidenceRegion(x,'exp')
% 设定参数默认值
if nargin == 1
    distribution = 'norm';
    alpha = 0.05;
elseif nargin == 2
    if ischar(alpha)
        distribution = alpha;
        alpha = 0.05;
    else
        distribution = 'norm';
    end
end
% 判断参数取值是否合适
if ~isscalar(alpha) || alpha>=1 || alpha<=0
    error('alpha的取值介于0和1之间')
end
if ~strncmpi(distribution,'norm',3) && ~strncmpi(distribution,'experience',3)
    error('分布类型只能是正态分布（''norm''）或经验分布（''experience''）')
end
% 检查数据维数是否正确
[m,n] = size(xdat);
p = min(m,n);  % 维数
if ~ismember(p,[1,2,3])
    error('应输入一维、二维或三维样本数据,并且样本容量应大于3')
end
% 把样本观测值矩阵转置，使得行对应观测，列对应变量
if m < n
    xdat = xdat';
end
xm = mean(xdat); % 均值
n = max(m,n);  % 观测组数
% 分情况绘制置信区域
switch p
    case 1    % 一维情形（置信区间）
        xstd = std(xdat); % 标准差
        if strncmpi(distribution,'norm',3)
            lo = xm - xstd*norminv(1-alpha/2,0,1); % 正态分布情形置信下限
            up = xm + xstd*norminv(1-alpha/2,0,1); % 正态分布情形置信上限
            %lo = xm - xstd*tinv(1-alpha/2,n-1); % 正态分布情形置信下限
            %up = xm + xstd*tinv(1-alpha/2,n-1); % 正态分布情形置信上限
            TitleText = 'Error Ellipsoid';
        else
            lo = prctile(xdat,100*alpha/2); % 经验分布情形置信下限
            up = prctile(xdat,100*(1-alpha/2)); % 经验分布情形置信上限
            TitleText = 'Error Ellipsoid';
        end
        % 对落入区间内外不同点用不同颜色和符号绘图
        xin = xdat(xdat>=lo & xdat<=up);
        xid = find(xdat>=lo & xdat<=up);
        plot(xid,xin,'.')
        hold on
        xout = xdat(xdat<lo | xdat>up);
        xid = find(xdat<lo | xdat>up);
        plot(xid,xout,'r+')
        h = refline([0,lo]);
        set(h,'color','k','linewidth',2)
        h = refline([0,up]);
        set(h,'color','k','linewidth',2)
        xr = xlim;
        yr = ylim;
        text(xr(1)+range(xr)/20,lo-range(yr)/20,'置信下限',...
            'color','g','FontSize',15,'FontWeight','bold')
        text(xr(1)+range(xr)/20,up+range(yr)/20,'置信上限',...
            'color','g','FontSize',15,'FontWeight','bold')
        xlabel('X in [m]')
        ylabel('Y in [m]')
        title(TitleText)
        hold off
    case 2    % 二维情形（置信椭圆）
        x = xdat(:,1);
        y = xdat(:,2);
        s = inv(cov(xdat));  % 协方差矩阵
        xd = xdat-repmat(xm,[n,1]);
        rd = sum(xd*s.*xd,2);
        if strncmpi(distribution,'norm',3)
            r = chi2inv(1-alpha,p);
            %r = p*(n-1)*finv(1-alpha,p,n-p)/(n-p)/n;
            TitleText = 'Error Ellipse';
        else
            r = prctile(rd,100*(1-alpha));
            TitleText = 'Error Ellipse';
        end
        plot(x(rd<=r),y(rd<=r),'.','MarkerSize',5)  % 画样本散点
        hold on
        plot(x(rd>r),y(rd>r),'r+','MarkerSize',5)  % 画样本散点
        plot(xm(1),xm(2),'k+');  % 椭圆中心
        h = ellipsefig(xm,s,r,1);  % 绘制置信椭圆
        xlabel('X in [m]')
        ylabel('Y in [m]')
        title(TitleText)
        hold off;
    case 3    % 三维情形（置信椭球）
        x = xdat(:,1);
        y = xdat(:,2);
        z = xdat(:,3);
        s = inv(cov(xdat));  % 协方差矩阵
        xd = xdat-repmat(xm,[n,1]);
        rd = sum(xd*s.*xd,2);
        if strncmpi(distribution,'norm',3)
            r = chi2inv(1-alpha,p);
            %r = p*(n-1)*finv(1-alpha,p,n-p)/(n-p)/n;
            TitleText = 'Error Ellipsoid';
        else
            r = prctile(rd,100*(1-alpha));
            TitleText = 'Error Ellipsoid';
        end
        plot3(x(rd<=r),y(rd<=r),z(rd<=r),'.','MarkerSize',5)  % 画样本散点
        hold on
        plot3(x(rd>r),y(rd>r),z(rd>r),'r+','MarkerSize',5)  % 画样本散点
        plot3(xm(1),xm(2),xm(3),'k+');  % 椭球中心
        h = ellipsefig(xm,s,r,2);  % 绘制置信椭球
        xlabel('X')
        ylabel('Y')
        zlabel('Z')
        color bar
        title(TitleText)
        hidden off;
        hold off;
end
end

function  h = ellipsefig(xc,P,r,tag)
% 画一般椭圆或椭球：(x-xc)'*P*(x-xc) = r
[V, D] = eig(P); 
if tag == 1
    aa = sqrt(r/D(1));
    bb = sqrt(r/D(4));
    t = linspace(0, 2*pi, 60);
    xy = V*[aa*cos(t);bb*sin(t)];  % 坐标旋转
    h = plot(xy(1,:)+xc(1),xy(2,:)+xc(2), 'k', 'linewidth', 2);
else
    aa = sqrt(r/D(1,1));
    bb = sqrt(r/D(2,2));
    cc = sqrt(r/D(3,3));
    [u,v] = meshgrid(linspace(-pi,pi,30),linspace(0,2*pi,30));
    x = aa*cos(u).*cos(v);
    y = bb*cos(u).*sin(v);
    z = cc*sin(u);
    xyz = V*[x(:)';y(:)';z(:)'];  % 坐标旋转
    x = reshape(xyz(1,:),size(x))+xc(1);
    y = reshape(xyz(2,:),size(y))+xc(2);
    z = reshape(xyz(3,:),size(z))+xc(3);
    h = mesh(x,y,z);  % 绘制椭球面网格图
end
end

