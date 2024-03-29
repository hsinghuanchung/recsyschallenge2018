#include<iostream>
#include<string>
#include<cstring>
#include<map>
#include<fstream>
#include<cstdlib>
#include<utility>
#include<queue>
#include<cmath>
#include<vector>
#include<pthread.h>
const int number=20;
struct temp_prior{
	double value;
	long long node;
	bool operator< (const temp_prior &j) const
	{
		return value < j.value; 
	}
};
class mapping{
	private:
		const int thread_num = number;
		std::map<std::string,long long> dic;
		std::vector<std::string> re;
		std::vector< std::vector<double> > arr;
		int size;
		std::pair<long long,long long> pt[number];
		std::vector< std::vector<int> > query_in;
	public:
		mapping();				
		mapping(const char *,const char*,int);
		void testing()
		{
			for(int i=0;i<5;i++)
				std::cout<<re[i]<<"-\n";
		}
		int query_int(std::string arr);		//use url to find index
		std::string query_url(int);		//use index to find url
		static void* find(void*);
		static void* che(void*);
		void goal(char*,int);
		void checking(int,char*,char*);
};

struct temp_type{
	std::vector<std::string>* a;
	std::vector<std::string>* b;
	int index;
	mapping* re;
};
