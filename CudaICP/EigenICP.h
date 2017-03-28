#include <Eigen/StdVector>

#define ROW 3
#define COL 217088

Eigen::Matrix4f icpBruteCPU(float *cld1, float *cld2, int *index); //Point-to-point ICP

Eigen::Matrix4f icpPTPCPU(float *cld1, float *cld2, float *normals, int *index); //Point-to-plane ICP

void comp_normal(float *data, float *normals); //Normal computation

Eigen::Matrix4f pose_from_6dof(Eigen::VectorXf par); //Computes transformation matrix from (roll,pitch,yaw,tx,ty,tz)