#ifndef __CF_CLUSTER_H__
#define __CF_CLUSTER_H__

#include "common.h"

class CF_Node;
class CF_Cluster;

using CF_Vector = std::vector<CF_Cluster>;
using CF_Vector_it = CF_Vector::iterator;
//ÿ�������ز���
//���У�N�������нڵ����Ŀ��LS��N���ڵ�����Ժͣ�SS��N���ڵ��ƽ���͡�
class CF_Cluster
{
public:
    long N;
    DataPoint LS;//valarray<data_t>,data_t=double;
    data_t SS;//data_t=double

    DataPoint X0;//��ֵ
    data_t R, D;

    CF_Node* child;

    CF_Cluster(const DataPoint& point);
    CF_Cluster(const CF_Vector& entries);
    CF_Cluster(CF_Node *node);

    bool operator ==(const CF_Cluster &rhs) const;

    void add(const CF_Cluster& entry);
    void remove(const CF_Cluster& entry);

    CF_Vector_it findClosest(CF_Vector& clusters) const;

    static std::pair<CF_Vector_it, CF_Vector_it> getTwoClosest(CF_Vector &clusters);
    static std::pair<CF_Vector_it, CF_Vector_it> getTwoFarthest(CF_Vector &clusters);

private:
    void updateMetrics();
};

#endif // __CF_CLUSTER_H__

