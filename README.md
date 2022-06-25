# Point-Cloud-Registration
2021/10/10:
This is the repository of the master thesis of Yimin Zhang.

# Background
High integrity localization is a fundamental task for an autonomous driving system. Standard localization approaches are usually accomplished by point cloud registration, which is often based on (recursive) least squares estimation, for example, using Kalman filters. However, due to the susceptibility of least squares minimization to outliers, it is not robust.

# Approach
In this work, we introduce a new approach of robust localization technique for self-driving vehicle. The Maximum Consensus Technique is generelized to the application of point cloud registration, which bases on a L0 loss. Therefore, it is much robuster than the classic least squares based approach, which uses the L2 loss. A comparision of L0, L1 and L2 loss functions is showed here

![image](https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/LossFunctions.PNG)

Specifically, we use the normal vectors of map point cloud to help allocate the car sensor scans. Point pairs are associated by comparing there distances inbetween. For a point pair of a map point and a measurement point in 2D case, an optimal translation ![](http://latex.codecogs.com/svg.latex?$t_{xy}$) needs to be determined to align two points, or in a point-to-plane registration task, to align the measurement point to the line depicted by the normal vector of the map point. The residual r is defined as the distance between the translated measurement point and the plane. Hence, we have the observation equation

![image](https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq1.PNG)

or

![image](https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq2.PNG)

where 

![image](https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq2.1.PNG)

is the observation term. Generelize the equation to all point pairs, which gives the observation equation in matrix form

![image](https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq3.PNG)

and the matrix A is the k x 2 matrix contains normal vectors of map points, which gives the indications of the estimation in the consensus set. The uncertainty of the measurement can be discribed by 

![image](https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq4.PNG)

For a qualitative comparision, we characterize the information of the observation by defining a score function

![image](https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq5.PNG)

and the best estimate is given by maximizing the score

![image](https://github.com/F95-6XL/Point-Cloud-Registration/blob/main/Images/eq6.PNG)


# Evaluation

corase GNSS position is used as search space origin
The title of this thesis is "Investigation of maximum consensus techniques for robust localization".
