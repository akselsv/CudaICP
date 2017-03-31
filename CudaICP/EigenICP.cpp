#include "EigenICP.h"

#include <Eigen/Dense>

#include <iostream>
#include <limits>
#include <cmath>

#define ROW 3
#define COL 217088

Eigen::Matrix4f icpBruteCPU(float *cld1, float *cld2, int *index) {

	int size = 1;
	for (int i = 0; i < COL; i++) {
		if (index[i] != -1) {
			size++;
		}
	}
	//read Data into Eigen matrices & compute centroids
	Eigen::MatrixXf a = Eigen::MatrixXf::Zero(ROW, size);
	Eigen::MatrixXf b = Eigen::MatrixXf::Zero(ROW, size);

	Eigen::Vector3f centroid_a = Eigen::Vector3f::Zero();
	Eigen::Vector3f centroid_b = Eigen::Vector3f::Zero();



	int it = 0;
	int count_a = 0;
	int count_b = 0;
	for (int i = 0; i < COL; i++) {
		if (index[i] != -1) {
			a(0, it) = cld1[i * 3 + 0];
			a(1, it) = cld1[i * 3 + 1];
			a(2, it) = cld1[i * 3 + 2];
			b(0, it) = cld2[index[i] * 3 + 0];
			b(1, it) = cld2[index[i] * 3 + 1];
			b(2, it) = cld2[index[i] * 3 + 2];

			centroid_a(0) += a(0, it);
			centroid_a(1) += a(1, it);
			centroid_a(2) += a(2, it);
			centroid_b(0) += b(0, it);
			centroid_b(1) += b(1, it);
			centroid_b(2) += b(2, it);
			it++;
		}
	}
	centroid_a = centroid_a / size;
	centroid_b = centroid_b / size;

	//Normalize
	for (int i = 0; i < size; i++) {
		a(0, i) -= centroid_a(0);
		a(1, i) -= centroid_a(1);
		a(2, i) -= centroid_a(2);
		b(0, i) -= centroid_b(0);
		b(1, i) -= centroid_b(1);
		b(2, i) -= centroid_b(2);
	}

	Eigen::Matrix3f SVD = Eigen::Matrix3f::Zero();
	//Create SVD
	for (int i = 0; i < size; i++) {
		SVD += b.block<3, 1>(0, i)*a.block<3, 1>(0, i).transpose(); //Compute SVD			
	}

	Eigen::JacobiSVD<Eigen::Matrix3f> USV(SVD, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix3f R = USV.matrixU()*USV.matrixV().transpose();

	/*if (R.determinant() < 0) { //check for reflection
		Eigen::Matrix3f Uminus = USV.matrixU();
		Uminus.block<3, 1>(0, 2) = -1 * USV.matrixU().block<3, 1>(0, 2);
		R = Uminus*USV.matrixV().transpose();
	}*/

	Eigen::Vector3f T = centroid_b.head(3) - R*centroid_a.head(3);

	Eigen::Matrix4f re = Eigen::Matrix4f::Identity();
	re.block<3, 3>(0, 0) = R;
	re.block<3, 1>(0, 3) = T;

	return re;
}

Eigen::Matrix4f icpPTPCPU(float *cld1, float *cld2, float *normals, int *index) {
	//Create A and b matrice *Ax-b*
	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(6, 6);
	Eigen::VectorXf b = Eigen::VectorXf::Zero(6);

	Eigen::Vector3f c;
	Eigen::Vector3f p;
	Eigen::Vector3f q;
	Eigen::Vector3f n;
	Eigen::VectorXf dum = Eigen::VectorXf::Zero(6);

	int it = 0;
	for (int i = 0; i < COL; i++) {
		if (index[i] != -1) {
			p(0) = cld1[i * 3 + 0]; p(1) = cld1[i * 3 + 1]; p(2) = cld1[i * 3 + 2];
			q(0) = cld2[index[i] * 3 + 0]; q(1) = cld2[index[i] * 3 + 1]; q(2) = cld2[index[i] * 3 + 2];
			n(0) = normals[index[i] * 3 + 0]; n(1) = normals[index[i] * 3 + 1]; n(2) = normals[index[i] * 3 + 2];
			c = p.cross(n);
			dum(0) = c(0); dum(1) = c(1); dum(2) = c(2); dum(3) = n(0); dum(4) = n(1); dum(5) = n(2);
			A += dum*dum.transpose();
			b(0) += c(0)*(p - q).dot(n); b(1) += c(1)*(p - q).dot(n); b(2) += c(2)*(p - q).dot(n);
			b(3) += n(0)*(p - q).dot(n); b(4) += n(1)*(p - q).dot(n); b(5) += n(2)*(p - q).dot(n);
		}
	}
	//Eigen::JacobiSVD<Eigen::MatrixXf> USV(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
	//std::cout << USV.matrixV*USV.singularValues().asDiagonal().inverse()*USV.matrixU().transpose()*b << std::endl;
	return pose_from_6dof(A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-b));
}

void comp_normal(float *data, float *normals) {
	int area = 5;
	Eigen::Matrix3d M;
	Eigen::Vector3d mean;
	int num = 0;

	for (int k = 0; k < COL; k++) {
		if (isfinite(data[k * 3])) {
			int a = k / 512; //this row
			int b = k % 512; //this col
			int num = 0;
			mean = Eigen::Vector3d::Zero();
			M = Eigen::Matrix3d::Zero();
			for (int i = -area; i <= area; i++) {
				for (int j = -area; j <= area; j++) {
					if (a + i >= 0 && a + i < 424 && b + j >= 0 && b + j < 512 && (i*i + j*j) <= (area + 1)*(area + 1)) { //this check makes sure we are within the limits of the 512x424 matrix and that we do a radius search instead of a square search
						int this_ind = (a + i) * 512 + b + j;
						if (isfinite(data[this_ind * 3])) {
							mean(0) += (double)data[this_ind * 3 + 0];
							mean(1) += (double)data[this_ind * 3 + 1];
							mean(2) += (double)data[this_ind * 3 + 2];
							num++;
						}
					}
				}
			}
			mean /= (double)num;
			for (int i = -area; i <= area; i++) {
				for (int j = -area; j <= area; j++) {
					if (a + i >= 0 && a + i < 424 && b + j >= 0 && b + j < 512 && (i*i + j*j) <= (area + 1)*(area + 1)) { //this check makes sure we are within the limits of the 512x424 matrix and that we do a radius search instead of a square search
						int this_ind = (a + i) * 512 + b + j;
						if (isfinite(data[this_ind * 3])) {
							double a = (double)data[this_ind * 3 + 0] - mean(0);
							double b = (double)data[this_ind * 3 + 1] - mean(1);
							double c = (double)data[this_ind * 3 + 2] - mean(2);
							M(0, 0) += a*a; M(0, 1) += b*a; M(0, 2) += c*a;
							M(1, 0) = M(0, 1); M(1, 1) += b*b; M(1, 2) += c*b;
							M(2, 0) = M(0, 2); M(2, 1) = M(1, 2); M(2, 2) += c*c;
						}
					}
				}
			}
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					M(i, j) /= (double)num;
				}
			}

			Eigen::EigenSolver<Eigen::Matrix3d> es(M);
			Eigen::Matrix3d mat = es.eigenvectors().real();
			Eigen::Vector3d V = es.eigenvalues().real();
			int smallest = 0;
			for (int i = 0; i < 3; i++) {
				if (V(i) < V(smallest)) {
					smallest = i;
				}
			}
			
			V = es.eigenvectors().col(smallest).real();
			normals[k * 3 + 0] = (float)V(0);
			normals[k * 3 + 1] = (float)V(1);
			normals[k * 3 + 2] = (float)V(2);
			if ((-normals[k * 3 + 0] * data[k * 3 + 0] - normals[k * 3 + 1] * data[k * 3 + 1] - normals[k * 3 + 2] * data[k * 3 + 2]) >= 0) {
				normals[k * 3 + 0] = -normals[k * 3 + 0];
				normals[k * 3 + 1] = -normals[k * 3 + 1];
				normals[k * 3 + 2] = -normals[k * 3 + 2];
			}
		}
		else {
			normals[k * 3 + 0] = 0;
			normals[k * 3 + 1] = 0;
			normals[k * 3 + 2] = 0;
		}
	}
}

Eigen::Matrix4f pose_from_6dof(Eigen::VectorXf par) {
	Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
	trans(0, 0) = cos(par(2))*cos(par(1));
	trans(0, 1) = -sin(par(2))*cos(par(0)) + cos(par(2))*sin(par(1))*sin(par(0));
	trans(0, 2) = sin(par(2))*sin(par(0)) + cos(par(2))*sin(par(1))*cos(par(0));
	trans(1, 0) = sin(par(2))*cos(par(1));
	trans(1, 1) = cos(par(2))*cos(par(0)) + sin(par(2))*sin(par(1))*sin(par(0));
	trans(1, 2) = -cos(par(2))*sin(par(0)) + sin(par(2))*sin(par(1))*cos(par(0));
	trans(2, 0) = -sin(par(1));
	trans(2, 1) = cos(par(1))*sin(par(0));
	trans(2, 2) = cos(par(1))*cos(par(0));
	trans(0, 3) = par(3);
	trans(1, 3) = par(4);
	trans(2, 3) = par(5);
	return trans;
}