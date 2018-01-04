
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
using namespace std;


#define pi 3.1415926535897932384626433832795

bool L1_flag=1;

//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
	double res=0;
    for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	res = sqrt(res);
	return res;
}

string version;
int relation_num,entity_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;


map<int,double> left_num,right_num;

class Train{

public:
	map<pair<int,int>, map<int,int> > train_set;
    void add(int x,int y,int z)
    {
        vec_head.push_back(x);
        vec_relation.push_back(z);
        vec_tail.push_back(y);
        train_set[make_pair(x,z)][y]=1;
    }
    void run(int n_in,double rate_in,double margin_in,int method_in)
    {
        dim = n_in;
        rate = rate_in;
        margin = margin_in;
        method = method_in;
        relation_vec.resize(relation_num);
		for (int i=0; i<relation_vec.size(); i++)
			relation_vec[i].resize(dim);
        entity_vec.resize(entity_num);
		for (int i=0; i<entity_vec.size(); i++)
			entity_vec[i].resize(dim);
        relation_tmp.resize(relation_num);
		for (int i=0; i<relation_tmp.size(); i++)
			relation_tmp[i].resize(dim);
        entity_tmp.resize(entity_num);
		for (int i=0; i<entity_tmp.size(); i++)
			entity_tmp[i].resize(dim);
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<dim; ii++)
                relation_vec[i][ii] = randn(0,1.0/dim,-6/sqrt(dim),6/sqrt(dim));
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<dim; ii++)
                entity_vec[i][ii] = randn(0,1.0/dim,-6/sqrt(dim),6/sqrt(dim));
            norm(entity_vec[i]);
        }


        bfgs();
    }

private:
    int dim,method;
    double loss;//loss function value
    double rate,margin;
    vector<int> vec_head,vec_tail,vec_relation;
    vector<vector<double> > relation_vec,entity_vec;
    vector<vector<double> > relation_tmp,entity_tmp;
    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
        for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }

    void bfgs()
    {
        loss=0;
        int nbatches=100;
        int nepoch = 1000;
        int batchsize = vec_head.size()/nbatches;
            for (int epoch=0; epoch<nepoch; epoch++)
            {

            	loss=0;
             	for (int batch = 0; batch<nbatches; batch++)
             	{
             		relation_tmp=relation_vec;
            		entity_tmp = entity_vec;
             		for (int k=0; k<batchsize; k++)
             		{
						int i=rand_max(vec_head.size());
						int j=rand_max(entity_num);
						double pr = 1000*right_num[vec_relation[i]]/(right_num[vec_relation[i]]+left_num[vec_relation[i]]);
						if (method ==0)
                            pr = 500;
						if (rand()%1000<pr)
						{
							while (train_set[make_pair(vec_head[i],vec_relation[i])].count(j)>0)
								j=rand_max(entity_num);
							train_kb(vec_head[i],vec_tail[i],vec_relation[i],vec_head[i],j,vec_relation[i]);
						}
						else
						{
							while (train_set[make_pair(j,vec_relation[i])].count(vec_tail[i])>0)
								j=rand_max(entity_num);
							train_kb(vec_head[i],vec_tail[i],vec_relation[i],j,vec_tail[i],vec_relation[i]);
						}
                		norm(relation_tmp[vec_relation[i]]);
                		norm(entity_tmp[vec_head[i]]);
                		norm(entity_tmp[vec_tail[i]]);
                		norm(entity_tmp[j]);
             		}
		            relation_vec = relation_tmp;
		            entity_vec = entity_tmp;
             	}
                cout<<"epoch:"<<epoch<<' '<<loss<<endl;
                FILE* f2 = fopen(("relation2vec."+version).c_str(),"w");
                FILE* f3 = fopen(("entity2vec."+version).c_str(),"w");
                for (int i=0; i<relation_num; i++)
                {
                    for (int ii=0; ii<dim; ii++)
                        fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                    fprintf(f2,"\n");
                }
                for (int i=0; i<entity_num; i++)
                {
                    for (int ii=0; ii<dim; ii++)
                        fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                    fprintf(f3,"\n");
                }
                fclose(f2);
                fclose(f3);
            }
    }
    double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<dim; ii++)
            	sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
        	for (int ii=0; ii<dim; ii++)
            	sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        return sum;
    }
    void gradient(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
        for (int ii=0; ii<dim; ii++)
        {

            double x = 2*(entity_vec[e2_a][ii]-entity_vec[e1_a][ii]-relation_vec[rel_a][ii]);
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel_a][ii]-=-1*rate*x;
            entity_tmp[e1_a][ii]-=-1*rate*x;
            entity_tmp[e2_a][ii]+=-1*rate*x;
            x = 2*(entity_vec[e2_b][ii]-entity_vec[e1_b][ii]-relation_vec[rel_b][ii]);
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel_b][ii]-=rate*x;
            entity_tmp[e1_b][ii]-=rate*x;
            entity_tmp[e2_b][ii]+=rate*x;
        }
    }
    void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
        double sum1 = calc_sum(e1_a,e2_a,rel_a);
        double sum2 = calc_sum(e1_b,e2_b,rel_b);
        if (sum1+margin>sum2)
        {
        	loss+=margin+sum1-sum2;
        	gradient( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
        }
    }
};

Train train;
void prepare()
{
    map<int,map<int,int> > left_entity,right_entity;
    char buf[100000];
    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/relation2id.txt","r");
	int id;
	while (fscanf(f1,"%s%d",buf,&id)==2)
	{
		string entity=buf;
		entity2id[entity]=id;
		id2entity[id]=entity;
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&id)==2)
	{
		string relation=buf;
		relation2id[relation]=id;
		id2relation[id]=relation;
		relation_num++;
	}
    FILE* f_kb = fopen("../data/train.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string head=buf;
        fscanf(f_kb,"%s",buf);
        string tail=buf;
        fscanf(f_kb,"%s",buf);
        string relation=buf;
        if (entity2id.count(head)==0)
        {
            cout<<"miss entity:"<<head<<endl;
        }
        if (entity2id.count(tail)==0)
        {
            cout<<"miss entity:"<<tail<<endl;
        }
        if (relation2id.count(relation)==0)
        {
            relation2id[relation] = relation_num;
            relation_num++;
        }
        left_entity[relation2id[relation]][entity2id[head]]++;
        right_entity[relation2id[relation]][entity2id[tail]]++;
        train.add(entity2id[head],entity2id[tail],relation2id[relation]);
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	left_num[i]=sum2/sum1;
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	right_num[i]=sum2/sum1;
    }
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
    fclose(f_kb);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    int method = 1;
    int n = 100;
    double rate = 0.001;
    double margin = 1;
    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-method", argc, argv)) > 0) method = atoi(argv[i + 1]);
    cout<<"size = "<<n<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;
    if (method)
        version = "bern";
    else
        version = "unif";
    cout<<"method = "<<version<<endl;
    prepare();
    train.run(n,rate,margin,method);
}

