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
	int node;
	bool operator< (const temp_prior &j) const
	{
		return value < j.value;
	}
};


class mapping{
	private:
		const int thread_num = number;
		std::map<std::string,int> dic;
		std::vector<std::string> re;
		std::vector< std::vector<double> > arr;
		int size;
		std::pair<int,int> pt[number];
	public:
		mapping();				
		mapping(const char *,const char*,int);
		int query_int(std::string arr);		//use url to find index
		std::string query_url(int);		//use index to find url
		static void* find(void*);
		void goal();
};

