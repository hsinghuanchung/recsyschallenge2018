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
struct temp{
	double value;
	int node;
	bool operator< (const temp&j) const
	{
		return value < j.value;
	}
};


class mapping{
	private:
		std::map<std::string,int> dic;
		std::vector<std::string> re;
		std::vector< std::vector<double> > arr;
		int size;
	public:
		mapping();				
		mapping(const char *,const char*,int);
		int query_int(std::string arr);		//use url to find index
		std::string query_url(int);		//use index to find url
		void find();
};

