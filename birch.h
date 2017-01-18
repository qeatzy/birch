#pragma once
#include "common.h"
#include "metrics.h"
#include "cftree.h"
#include "cftreebuilder.h"
#include <time.h>
#include <cmath>
#include "readcsv.h"
#include"standardnaming.h"
double E_distance(std::vector<double> X,std::vector<double> Y, int dim)
{
	double dis = 0;
	for (int i = 0; i < dim; i++)
	{
		dis += (X[i] - Y[i])*(X[i] - Y[i]);
	}
	return std::sqrt(dis);
}
int findcluster(std::vector<double> X, Datastruct center, int k_clustering, int dim)
{
	int Minx = 9999999;
	int pos = -1;
	for (int i = 0; i < k_clustering; i++)
	{
		double tmp_dis = E_distance(X, center[i], dim);
		if (Minx > tmp_dis)
		{
			pos = i;
			Minx = tmp_dis;
		}
	}
	return pos;
}
int *Datalabel(Datastruct &data, Datastruct &center, int k_clustering, int len, int dim)
{
	int *label = new int[len]();
	for (int i = 0; i < len; i++)
	{
		label[i] = findcluster(data[i], center, k_clustering, dim);
	}
	return label;
}

std::vector<CF_Vector> distrKMeans(const CF_Vector &entries, size_t dim, int k_clustering)
{
	int localEntries = entries.size(),
		globalEntries = 0;
	//聚类中心数量
	int k = k_clustering;
	int data_size = entries.size();

	std::vector<DataPoint> centroidSeeds(k, DataPoint((data_t)0, dim));
	//得到所有的数据
	auto allData = CF_Cluster(entries);
	//初始化得到K个聚类中心，每个维度上的值
	int *random_k = new int[k];
	int *arry_index = new int[data_size];
	std::fill_n(arry_index, data_size, 0);
	srand((int)time(0));
	for (int i = 0; i < k; i++)
	{
		int tmp = rand() % data_size;
		while (arry_index[tmp] != 0)
		{
			tmp = rand() % data_size;
		}
		random_k[i] = tmp;
		arry_index[tmp] = 1;
	}
	for (int j = 0; j < k; ++j)
	{
		centroidSeeds[j] = entries[random_k[j]].X0;
	/*	for (int i = 0; i < dim; i++)
		{
			cout << "X0:" << entries[random_k[j]].X0[i] << " " << "LS:" << entries[random_k[j]].LS[i] << "N:" << entries[random_k[j]].N << endl;
		}
		cout << "ID" << " " << random_k[j] << endl;
	*/
	}
//	cout << "RAND_MAX * allData.R:" << RAND_MAX * allData.R << endl;
	//输出初始化的聚类中心
/*
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < dim; j++)
			cout << centroidSeeds[i][j] << " ";
		cout << endl;
	}
*/
	//得到聚类中心
	CF_Vector centroids;
	for (int j = 0; j < k; ++j)
		centroids.emplace_back(centroidSeeds[j]);
	centroidSeeds.clear();

	std::vector<CF_Vector> clusters(k);
	data_t localMSE = std::numeric_limits<data_t>::max(),
		MSE;
	std::vector<int> localN(k);
	std::vector<DataPoint> localSum(k, DataPoint((data_t)0, dim)), globalSum(k, DataPoint((data_t)0, dim));
	do
	{
		MSE = localMSE;
		localMSE = 0;
		for (int j = 0; j < k; ++j)
		{
			localN[j] = 0;
			localSum[j] = DataPoint((data_t)0, dim);
			clusters[j].clear();
		}
		for (size_t i = 0; i < entries.size(); ++i)
		{
			auto closest = entries[i].findClosest(centroids);
			size_t ind = closest - centroids.begin();

			clusters[ind].push_back(entries[i]);
			localN[ind] += entries[i].N;
			localSum[ind] += entries[i].LS;

			data_t dist = getDistance(entries[i], *closest);
			localMSE += dist*dist;
		}
		for (int j = 0; j < k; ++j)
		{
			centroids[j] = CF_Cluster(localSum[j] / (data_t)localN[j]);
			//            LOG_DEBUG("New %d cluster: %s", j + 1, pointToString(centroids[j].X0).c_str());
		}
	} while (localMSE < MSE);

	return  clusters;
}
//得到聚类中心
Datastruct getClusteringCenter(std::vector<CF_Vector> clusters,int data_dim,int k_clustering)
{
	int i, j,z,sum;
	Datastruct center(k_clustering, std::vector<double>(data_dim, 0));
	for (i = 0; i < k_clustering; i++)
	{
		sum = 0;
		for (j = 0; j < clusters[i].size(); j++)
		{
			sum += clusters[i][j].N;
			for (z = 0; z < data_dim; z++)
				center[i][z] += clusters[i][j].LS[z];
		}
		for (j = 0; j < data_dim; j++)
			center[i][j] = center[i][j] / sum;
	}
/*	for (i = 0; i < k_clustering; i++)
	{
		for (j = 0; j < 3; j++)
			cout << setprecision(4) << center[i][j] << " ";
		cout << endl;
	}*/
	return center;
}
//find the best k
//work=1：工况划分，work=0:运行方式划分
int find_best_k(Datastruct &data, int work,int data_dim)
{
	int k,i,j;
	int k_clustering = 0;
	int data_len = data.size();
	double stand_dis = DBL_MAX;
	if (work)
	{
		for (k = 5; k < 15; k++)
		{
			double dis_everydata = 0;
			CF_TreeBuilder machine_condition_treeBuilder(data_len, data_dim, std::log(data_len), 0, std::sqrt(data_len), std::log(data_len));
			for (i = 0; i < data_len; i++)
			{
				DataPoint dataPoint(data_dim);
				for (j = 0; j < data_dim; j++)
				{
					dataPoint[j] = data[i][j];
				}
				machine_condition_treeBuilder.addPointToTree(dataPoint);
			}
			auto machine_condition_entries = machine_condition_treeBuilder.getAllLeafEntries();
			auto clusters = distrKMeans(machine_condition_entries, data_dim, k);
			Datastruct center = getClusteringCenter(clusters, data_dim, k);
			for (i = 0; i < data_len; i++)
			{
				int clustering_id = findcluster(data[i], center, k, data_dim);
				dis_everydata += E_distance(data[i], center[clustering_id], data_dim);
			}
			std::cout << log10(dis_everydata) << std::endl;
			if (stand_dis > dis_everydata)
			{
				stand_dis = dis_everydata;
				k_clustering = k;
			}
		}
	}
	else
	{
		for (k = 2; k < 5; k++)
		{
			double dis_everydata = 0;
			bool judge_null = 0;
			CF_TreeBuilder machine_condition_treeBuilder(data_len, data_dim, std::log(data_len), 0, std::sqrt(data_len), std::log(data_len));
			for (i = 0; i < data_len; i++)
			{
				DataPoint dataPoint(data_dim);
				for (j = 0; j < data_dim; j++)
				{
					dataPoint[j] = data[i][j];
				}
				machine_condition_treeBuilder.addPointToTree(dataPoint);
			}
			auto machine_condition_entries = machine_condition_treeBuilder.getAllLeafEntries();
			auto clusters = distrKMeans(machine_condition_entries, data_dim, k);
			Datastruct center = getClusteringCenter(clusters, data_dim, k);
			//避免某一个类出现无样本的情况
			int *label = Datalabel(data, center, k, data_len, data_dim);
			int *num = new int[k]();
			for (i = 0; i < data_len; i++)
				num[label[i]]++;
			for (i = 0; i < k; i++)
				if (num[i] == 0)
					judge_null = 1;
			if (judge_null)
				return k_clustering;
			//如果都有样本
			for (i = 0; i < data_len; i++)
			{
				int clustering_id = findcluster(data[i], center, k, data_dim);
				dis_everydata += E_distance(data[i], center[clustering_id], data_dim);
			}
			if (stand_dis > dis_everydata)
			{
				stand_dis = dis_everydata;
				k_clustering = k;
			}
		}
	}
	return k_clustering;
}