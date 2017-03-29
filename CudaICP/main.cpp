#include <fstream>
#include <iostream> //Only for printig Eigen stuff
#include <sstream>
#include <ctime>
#include "EigenICP.h"
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <cuda_runtime.h>

#include "kernel.h"

#define ROW 3 //dimension for each point
#define COL 217088 //Number of points

void readFile(float *data, std::string name); //Read file named "name" to "float *data"
void writeFile(float *data, int i); //Write "float *data" to file named "outcloud" + i + ".txt
void writeFile2(float *valse, float*balse, int *indices); //Write corresponding points indicated by "int *indices" to file
Eigen::Matrix3f rot_from_RPY(Eigen::Vector3f rpy); //Calculate rotation matric from roll-pitch-yaw angles
Eigen::Vector3f RPY_from_rot(Eigen::Matrix3f rot); //Calculate roll-pitch-yaw angles from rotation matrix

int main() {
	float *normals_GPU = new float[ROW*COL];
	float *model = new float[ROW*COL];
	std::string name;
	name = "Cloud_uncompressed" + std::to_string(0) + ".txt";
	readFile(model, name); //Read file to "model"
	testing(model,normals_GPU);

	for (int i = 0; i < COL; i++) {
		printf("%f\t%f\t%f\n", normals_GPU[i * 3 + 0], normals_GPU[i * 3 + 1], normals_GPU[i * 3 + 2]);
	}

	delete[] normals_GPU; normals_GPU = nullptr;
	delete[] model; model = nullptr;



	/*int icp_iteratons = 100; //Number of ICP iterations
	int dataset_start = 30; //Dataset start number. Format: "name" + number + ".txt"
	int dataset_stop = 31; //Dataset stop number. Format: "name" + number + ".txt"


	long startTime;
	long endTime;
	float *model = new float[ROW*COL];
	float *target = new float[ROW*COL];
	float *normals = new float[ROW*COL];
	int* cpu_ptrclosest = new int[COL];
	std::string name;
	std::ofstream outfile;
	std::ofstream transfile;
	Eigen::MatrixXf Temp = Eigen::MatrixXf::Zero(ROW + 1, COL);
	Eigen::Matrix4f accuICPTrans = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f absTrans = Eigen::Matrix4f::Identity();

	outfile.open("performance.txt"); //File for storing perfromance of ICP
	transfile.open("transforms.txt"); //File for storing found transforms

	name = "Cloud_uncompressed" + std::to_string(dataset_start) + ".txt";
	readFile(model, name); //Read file to "model"

	//Start registration
	for (int i = dataset_start+1; i < dataset_stop+1; i++) {
		//Previous model becomes target
		float *temp = target;
		target = model;
		model = temp;

		//Read new model data
		name = "Cloud_uncompressed" + std::to_string(i) + ".txt";
		readFile(model, name); //Read file to "model"

		//Store data in Eigen matrix
		for (int j = 0; j < COL; j++) {
			Temp(0, j) = model[j * 3 + 0];
			Temp(1, j) = model[j * 3 + 1];
			Temp(2, j) = model[j * 3 + 2];
			Temp(3, j) = 1;
		}

		//Transform to current pose..
		Temp = absTrans*Temp;

		//..and update model data
		for (int j = 0; j < COL; j++) {
			model[j * 3 + 0] = Temp(0, j);
			model[j * 3 + 1] = Temp(1, j);
			model[j * 3 + 2] = Temp(2, j);
		}

		//Compute normals
		startTime = clock();
		comp_normal(target,normals);
		endTime = clock();
		printf("Normal computation took: %ld ms \n", endTime - startTime);

		//iterate ICP several times
		for (int q = 0; q < icp_iteratons; q++) {
			int corr = 0;

			//Find NN on GPU
			float executiontime = run_procedure(model, target, cpu_ptrclosest);			

			//Calculate ICP transfrom from NN
			//Eigen::Matrix4f trans = icpBruteCPU(model, target, cpu_ptrclosest); //Point-to-Point ICP
			Eigen::Matrix4f trans = icpPTPCPU(model, target, normals, cpu_ptrclosest); //Point-to-Plane ICP

			//Update accumulated transformation
			accuICPTrans = trans*accuICPTrans;

			//Update point cloud
			Temp = trans*Temp;
			for (int j = 0; j < COL; j++) {
				model[j * 3 + 0] = Temp(0, j);
				model[j * 3 + 1] = Temp(1, j);
				model[j * 3 + 2] = Temp(2, j);
			}

			//Calculate ICP-RMS error and count the number of correspondences
			float rms_err = 0;			
			for (int j = 0; j < COL; j++) {
				if (cpu_ptrclosest[j] != -1) {
					rms_err += (model[j * 3 + 0] - target[cpu_ptrclosest[j] * 3 + 0])*(model[j * 3 + 0] - target[cpu_ptrclosest[j] * 3 + 0]) +
						(model[j * 3 + 1] - target[cpu_ptrclosest[j] * 3 + 1])*(model[j * 3 + 1] - target[cpu_ptrclosest[j] * 3 + 1]) +
						(model[j * 3 + 2] - target[cpu_ptrclosest[j] * 3 + 2])*(model[j * 3 + 2] - target[cpu_ptrclosest[j] * 3 + 2]);
					corr++;
				}
			}
			rms_err = sqrt(rms_err) / corr;

			//Write Performance to file
			outfile << RPY_from_rot(accuICPTrans.block<3, 3>(0, 0)).transpose() << " " << accuICPTrans.block<3, 1>(0, 3).transpose() << " " << corr << " " << rms_err << " " << executiontime << "\n";
		}

		//update transformation and write to file
		absTrans = accuICPTrans*absTrans;
		transfile << absTrans << "\n";

		//reset interframe transfomration
		accuICPTrans = Eigen::Matrix4f::Identity();
	}

	//Close and terminate
	transfile.close();
	outfile.close();	
	delete[] normals; normals = nullptr;
	delete[] model; model = nullptr;
	delete[] target; target = nullptr;
	delete[] cpu_ptrclosest; cpu_ptrclosest = nullptr;

	*/
	system("pause");
	return 0;
}

Eigen::Matrix3f rot_from_RPY(Eigen::Vector3f rpy) {
	Eigen::Matrix3f rotz = Eigen::Matrix3f::Identity();
	rotz(0, 0) = cosf(rpy(2));
	rotz(1, 0) = sinf(rpy(2));
	rotz(0, 1) = -sinf(rpy(2));
	rotz(1, 1) = cosf(rpy(2));

	Eigen::Matrix3f roty = Eigen::Matrix3f::Identity();
	roty(0, 0) = cosf(rpy(1));
	roty(2, 2) = cosf(rpy(1));
	roty(0, 2) = sinf(rpy(1));
	roty(2, 0) = -sinf(rpy(1));

	Eigen::Matrix3f rotx = Eigen::Matrix3f::Identity();
	rotx(1, 1) = cosf(rpy(0));
	rotx(2, 2) = cosf(rpy(0));
	rotx(2, 1) = sinf(rpy(0));
	rotx(1, 2) = -sinf(rpy(0));

	return rotz*roty*rotx;
}

Eigen::Vector3f RPY_from_rot(Eigen::Matrix3f R) {
	Eigen::Vector3f a = Eigen::Vector3f::Zero();
	a(2) = atan2f(R(1, 0), R(0, 0));
	a(1) = atan2f(-R(2, 0), sqrtf(powf(R(2, 1), 2) + powf(R(2, 2), 2)));
	a(0) = atan2f(R(2, 1), R(2, 2));
	return a;
}

void writeFile2(float *valse, float *balse, int *indices) {
	std::ofstream outfile;
	std::ostringstream oss;
	oss << "corr1.txt";
	outfile.open(oss.str());

	for (int i = 0; i < COL; i++) {
		if (indices[i] != -1) { // != -std::numeric_limits<float>::infinity()) {
			outfile << valse[i * 3 + 0] << " " << valse[i * 3 + 1] << " " << valse[i * 3 + 2] << "\n";
		}
	}
	outfile.close();
	std::ostringstream osss;
	osss << "corr2.txt";
	outfile.open(osss.str());
	for (int i = 0; i < COL; i++) {
		if (indices[i] != -1) { // != -std::numeric_limits<float>::infinity()) {
			outfile << balse[indices[i] * 3 + 0] << " " << balse[indices[i] * 3 + 1] << " " << balse[indices[i] * 3 + 2] << "\n";
		}
	}
	outfile.close();
}

void writeFile(float *valse, int num) {
	std::ofstream outfile;
	std::ostringstream oss;
	oss << "outcloud" << num << ".txt";
	outfile.open(oss.str());

	for (int i = 0; i < COL; i++) {
		if (isfinite(valse[i * 3])) { // != -std::numeric_limits<float>::infinity()) {
			outfile << valse[i * 3 + 0] << " " << valse[i * 3 + 1] << " " << valse[i * 3 + 2] << "\n";
		}
	}

	outfile.close();
}

void readFile(float *data, std::string name) {
	std::fstream in;
	in.open(name);
	if (in.fail()) {
		printf("error loading .txt file reading\n");
		return;
	}
	std::string line;
	std::string value;
	int i = 0;
	while (std::getline(in, line)) {
		std::stringstream ss(line);
		while (ss >> value) {
			if (value == "-inf") {
				data[i] = -std::numeric_limits<float>::infinity();
			}
			else {
				data[i] = std::stof(value);
			}
			i++;
		}
	}
	in.close();
}