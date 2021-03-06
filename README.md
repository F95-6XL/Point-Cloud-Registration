# Point-Cloud-Registration
2021/10/10:
*This is the repository of the master thesis of Yimin Zhang from Leibniz University of Hannover, Germany.*

# Background
High integrity localization is a fundamental task for an autonomous driving system. Standard localization approaches are usually accomplished by point cloud registration, which is often based on (recursive) least squares estimation, for example, using Kalman filters. However, due to the susceptibility of least squares minimization to outliers, it is not robust.

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/PointCloudForLocalization.png" width="480">
</div>

# Approach
In this work, we introduce a new approach of robust localization technique for self-driving vehicle. The Maximum Consensus Technique is generelized to the application of point cloud registration, which bases on a L0 loss. Therefore, it is much robuster than the classic least squares based approach, which uses the L2 loss. A comparision of L0, L1 and L2 loss functions is showed here

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/LossFunctions.PNG" width="480">
</div>

Specifically, we use the normal vectors of map point cloud to help allocate the car sensor scans. Point pairs are associated by comparing there distances inbetween. For a point pair of a map point and a measurement point in 2D case, an optimal translation *t* needs to be determined to align two points, or in a point-to-plane registration task, to align the measurement point to the line depicted by the normal vector of the map point. The residual r is defined as the distance between the translated measurement point and the plane. Hence, we have the observation equation

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq1.PNG" height="60">
</div>

or

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq2.PNG" height="35">
</div>

where 

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq2.1.PNG" height="35">
</div>

is the observation term. 

Generelize the equation to all point pairs, we have the observation equation in matrix form

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq3.PNG" height="30">
</div>

and the matrix **A** is the *k x 2* matrix contains normal vectors of map points. And here comes the difference. Instead of minimizing the residual ***r*** in common approaches, we focus on the information provided by the measurements. Matrix **A**  contains all "votes" of the map points in the consensus set, which then discribes the indications of the measurement in each direction. Hence, the uncertainty of the measurement can be discribed by 

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq4.PNG" height="40">
</div>

with an omitted constant (co-factor). 

For a qualitative comparision, we characterize the information of the observation by defining a score function

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq5.PNG" height="60">
</div>

and the best estimate is given by maximizing the score

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq6.PNG" height="60">
</div>


# Evaluation

The approach is tested in an inner city area characterized by a dense building structures. A corase GNSS position is used as search space origin. The search space is discritized with a user defined metric *e*. Here we choose the grid size to be *0.02m*.

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/Map.PNG" width="680">
</div>

The test set consists of 1915 epoches with a time interval of 0.1 second between two epoches, which were acquired by a Velodyne VLP-16 Lidar. Below also shows some of the measurement scans.

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/Measurements.PNG" width="780">
</div>

The approach is then evaluated on the test data set. The error distribution over all 1915 epoches are displayed below

<div align=center>
<img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/Error in X.PNG" width="280"><img src="https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/Error in Y.PNG" width="260">
</div>

The result shows an average localization error of less than *0.01m* in both direction. And the maximum error recorded is *0.3m*. The Evaluation shows a generally high intergriy localization results.

# About the code

The Approach is implemented in Python and is not yet optimized. 

From search dimension perspective, We support 4 modes of search, namely ***xy-translation***, ***xy-translation + z-rotation***, ***xyz-translation***, ***xyz-translation + z-rotation***.

From search approach perspective, we enable two maximum consensus based approach of search:

1. Use count of point matches as score to build up accumulator, which is called version 1.0 
2. Use trace of covariance matrix as score to build up accumulator, which is called version 1.0 

Some other setup are implemented as run configuration, e.g., ***remove ground points***, ***using weighted score***, ***downsampling of measurements***.

Besides, an ICP step using the outcome of maximum consensus search is implemented to further conduct a fine and continious search.
