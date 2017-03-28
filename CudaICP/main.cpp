//#include <stdio.h>
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

void readFiles(float *valse, float *balse);
void readFile(float *data, std::string name);
void writeFile(float *valse, int i);
void writeFile2(float *valse, float*balse, int *indices);
Eigen::Matrix3f rot_from_RPY(Eigen::Vector3f rpy);
Eigen::Vector3f RPY_from_rot(Eigen::Matrix3f rot);

int main() {
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
	bool overshoot = false;

	outfile.open("performance.txt");
	transfile.open("transforms.txt");

	name = "Cloud_uncompressed" + std::to_string(30) + ".txt";
	readFile(model, name);
	for (int i = 31; i < 32; i++) {
		float *temp = target;
		target = model;
		model = temp;
		name = "Cloud_uncompressed" + std::to_string(i) + ".txt";
		readFile(model, name);



		for (int j = 0; j < COL; j++) {
			Temp(0, j) = model[j * 3 + 0];
			Temp(1, j) = model[j * 3 + 1];
			Temp(2, j) = model[j * 3 + 2];
			Temp(3, j) = 1;
		}

		//transform to current pose
		Temp = absTrans*Temp;

		//Update point cloud
		for (int j = 0; j < COL; j++) {
			model[j * 3 + 0] = Temp(0, j);
			model[j * 3 + 1] = Temp(1, j);
			model[j * 3 + 2] = Temp(2, j);
		}

		//iterate ICP several times

		//Compute normals
		startTime = clock();
		comp_normal(target,normals);
		endTime = clock();
		printf("Normal computation tool: %ld ms \n", endTime - startTime);

		for (int q = 0; q < 100; q++) {
			int corr = 0;

			//Find NN
			startTime = clock();
			float executiontime = run_procedure(model, target, cpu_ptrclosest);
			endTime = clock();
			//printf("GPU time: %ld ms\n", endTime - startTime);
			//printf("Actual time: %f ms\n", executiontime);
			//printf("iteration number %i\n", q + 1);

			//Calculate ICP transfrom from NN
			//Eigen::Matrix4f trans = icpBruteCPU(model, target, cpu_ptrclosest);
			Eigen::Matrix4f trans = icpPTPCPU(model, target, normals, cpu_ptrclosest);

			//Test overshooting transform
			if (overshoot) {
				Eigen::Vector3f EA = RPY_from_rot(trans.block<3, 3>(0, 0))*1.6;
				Eigen::Matrix3f Rot = rot_from_RPY(EA);
				std::cout << EA << std::endl;
				std::cout << Rot << std::endl;
				std::cout << trans.block<3, 3>(0, 0) << std::endl;
				trans.block<3, 3>(0, 0) = Rot;
				trans.block<3, 1>(0, 3) = trans.block<3, 1>(0, 3)*1.6;
			}

			//Update accumulated transformation
			accuICPTrans = trans*accuICPTrans;

			//Update point cloud
			Temp = trans*Temp;
			for (int j = 0; j < COL; j++) {
				model[j * 3 + 0] = Temp(0, j);
				model[j * 3 + 1] = Temp(1, j);
				model[j * 3 + 2] = Temp(2, j);
			}

			float rms_err = 0;
			//Calculate ICP-RMS error
			for (int j = 0; j < COL; j++) {
				if (cpu_ptrclosest[j] != -1) {
					rms_err += (model[j * 3 + 0] - target[cpu_ptrclosest[j] * 3 + 0])*(model[j * 3 + 0] - target[cpu_ptrclosest[j] * 3 + 0]) +
						(model[j * 3 + 1] - target[cpu_ptrclosest[j] * 3 + 1])*(model[j * 3 + 1] - target[cpu_ptrclosest[j] * 3 + 1]) +
						(model[j * 3 + 2] - target[cpu_ptrclosest[j] * 3 + 2])*(model[j * 3 + 2] - target[cpu_ptrclosest[j] * 3 + 2]);
					corr++;
				}
			}
			rms_err = sqrt(rms_err) / corr;
			//printf("Error: %f\n Correspondenses: %i\n", rms_err,corr);
			//Write Performance to file
			outfile << RPY_from_rot(accuICPTrans.block<3, 3>(0, 0)).transpose() << " " << accuICPTrans.block<3, 1>(0, 3).transpose() << " " << corr << " " << rms_err << " " << executiontime << "\n";
		}
		absTrans = accuICPTrans*absTrans;
		transfile << absTrans << "\n";
		printf("%i\n", i);
		std::cout << absTrans << std::endl;
		accuICPTrans = Eigen::Matrix4f::Identity();
		

	}
	transfile.close();
	outfile.close();
	//writeFile(valse, 4); //Cloud1 transformed = output4.txt
	//writeFile2(valse, balse, cpu_ptrclosest); //write correspondenses
	
	delete[] normals; normals = nullptr;
	delete[] model; model = nullptr;
	delete[] target; target = nullptr;
	delete[] cpu_ptrclosest; cpu_ptrclosest = nullptr;


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

void readFiles(float *valse, float *balse) { //[ROW*COL], float balse[ROW*COL]) {
	std::fstream in;
	in.open("Cloud1.txt");
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
				valse[i] = -std::numeric_limits<float>::infinity();
			}
			else {
				valse[i] = std::stof(value);
			}
			i++;
		}
	}
	in.close();
	in.open("Cloud2.txt");
	if (in.fail()) {
		printf("error loading .txt file reading\n");
		return;
	}
	i = 0;
	while (std::getline(in, line)) {
		std::stringstream sss(line);
		while (sss >> value) {
			if (value == "-inf") {
				balse[i] = -std::numeric_limits<float>::infinity();
			}
			else {
				balse[i] = std::stof(value);
			}
			i++;
		}
	}
	in.close();
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