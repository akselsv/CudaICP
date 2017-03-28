#include <Eigen/StdVector>

#define ROW 3
#define COL 217088

class EigenICP
{
public:
	EigenICP(float* val2, float *val1);
	~EigenICP();
	//Matrices to hold the data
	Eigen::MatrixXf pts1 = Eigen::MatrixXf::Zero(ROW, COL);
	Eigen::MatrixXf pts2 = Eigen::MatrixXf::Zero(ROW, COL);

	//Closest point correspondences
	Eigen::VectorXi corr = Eigen::VectorXi::Zero(COL);

	//Rotation matrix initialized to identity
	Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
	//Translation vector initialized to zero
	Eigen::Vector3f T = Eigen::Vector3f::Zero();

	//Matrix to hold SVD
	Eigen::Matrix3f SVD = Eigen::Matrix3f::Zero();

	//centroids
	Eigen::Vector3f centroid1;
	Eigen::Vector3f centroid2;

	void iterateICP();
	

private:
	int counter;
	float *ker_pts1;
	float *ker_pts2;
	int *ker_closest;
	void getNNfromKernel();
	void readData(float *cld1, float *cld2);
	void updateCorrespondences();
	void applyTransformation();
	void calculateTransformation();
	void computeCentroids();
	//Normal computation
	
};


Eigen::Matrix4f icpBruteCPU(float *cld1, float *cld2, int *index);
Eigen::Matrix4f icpPTPCPU(float *cld1, float *cld2, float *normals, int *index);
//Normal computation
void comp_normal(float *data, float *normals);
Eigen::Matrix4f pose_from_6dof(Eigen::VectorXf par);