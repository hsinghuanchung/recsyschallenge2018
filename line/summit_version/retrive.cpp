#include"retrive.h"
using namespace std;

mapping::mapping()
{
	mapping("./../../data/mapping","./../../data/out_result",64);
}
mapping::mapping(const char *file_name,const char *file,int size)
{
	//open the file here and make it into file
	this->size = size;
	string key_word;

	ifstream in(file_name); 
	string temp;
	int index,num,i,count;
	size_t pt;
	string number;


	printf("%s\n",file_name);

	getline(in,temp);
	getline(in,temp);

	while(getline(in,temp))
	{
		//cout<<index<<endl;
		index = temp.rfind(' ');
				
		num = atoi(&temp.c_str()[index+1]);	
	    temp.erase(temp.begin()+index,temp.end());

		re.push_back(temp);

		//check if it is title or track


		if(temp.find("spotify:track:"))
		{
			if( dic.find(temp) == dic.end() )
				dic[temp] = num;
			else
				cout<<"it seems to be some error\n";	
		}
		else
		{
			i=0;
			while(1)
			{
				key_word = "";
				while(1)
				{
					if('0'<=temp[i] && temp[i]<='9')
						key_word += temp[i];
					else if('a'<=temp[i] && temp[i]<='z')
						key_word += temp[i];
					else if('A'<=temp[i] && temp[i]<='Z')
						key_word += temp[i]-32;
					else if(temp[i]=='_')
						key_word += temp[i];
					else if(temp[i]=='\'')
						key_word += temp[i];
					else if(temp[i]==' ')
					{
						i++;
						break;
					}		
					i++;
					if(i==temp.size())
						break;
				}

				if(key_word[ key_word.size()-1 ] == 's' && key_word[ key_word.size()-1 ] == '\'' )
					key_word.erase(key_word.end()-2,key_word.end());

				if(i==temp.size())//last one11
				{
					pt = key_word.rfind('_');
					if(pt == 0 )
						break;
					key_word = key_word.substr(0,pt);
				}	

				count = 0;
				while(1)
				{	
					pt = key_word.rfind(' ');
					number = key_word+" "+to_string(count);	
					
					if( dic.find(number) == dic.end() )
					{
						dic[number] = num;
						break;
					}
					else
					{
						count++;	
					}	
				}
				if(i==temp.size())
					break;	
			}
		}
	}

	in.close();

	string::size_type s1,s2;
	
	double val;
	char size_num[128];
	sprintf(size_num,"%s%d",file,size);
	
		
	in.open(size_num);
	vector<double> data;
	
	arr.clear();
	while(getline(in,temp))
	{
		data.clear();
		index = stoi(temp,&s1);
		for(int i=0 ; i<this->size ; i++)
		{
			val = stod(temp.substr(s1),&s2);
			s1 += s2;  
			data.push_back(val);
		}
		arr.push_back(data);
	}
	this->testing();
}
int  mapping::query_int(string arr)
{
	if(dic.find(arr) == dic.end())
		return -1;

	return this->dic[arr];
}
string mapping::query_url(int index)
{
	return this->re[index];
}

void mapping::goal(char* in_file,int range=-1)	//input the query file here
{
	int now=0,index,pt;
	
	ifstream in(in_file); 
	string temp;
	vector<int> qu;

	getline(in,temp);
	while(in)
	{
		//first deal with title
		//for the same word there might be many version, we simply took its average



		
		//second deal with the track
		getline(in,temp);
		now=0;
		qu.clear();
		while(1)
		{
			pt = temp.find_first_of(" ",now);
			index = this->query_int(temp.substr(now,pt-now));	
			
			if(index==-1)	//don't put in it doesn't happen before
			{
			}
			else
				qu.push_back(index);
			now = pt+1;

			if(pt == temp.size())
				break;
		}
		this->query_in.push_back(qu);
		if(range>0)
			range--;
		if(range==0)
			break;
	}

	int ss=query_in.size()/this->thread_num;
	pthread_t thread[this->thread_num];
	pair<mapping*,int> parm[this->thread_num];
	void *res;

	for(int i=0;i<this->thread_num;i++)
	{
		this->pt[i].first = i*ss;
		this->pt[i].second =(i+1)*ss;

		if( i == (this->thread_num-1))
			this->pt[i].second = query_in.size();

		thread[i] = i;

		parm[i].first = this;
		parm[i].second = i;

		pthread_create(&thread[i],NULL,mapping::find,(void*)&parm[i]);
	}
	for(int i=0;i<this->thread_num;i++)
		pthread_join(thread[i],&res);
}

void* mapping::find(void *parm)
{
	pair<mapping*,int> *temp = (pair<mapping*,int>*) parm;
	string url;

	double one,all;

	temp_prior h; 
	long long i,j,l;
	int k;

	char file[128]="ans";
	sprintf(file,"/mnt/data/recsys_spotify/line_data/ans_%d_[%lld_%lld].out",temp->first->size,temp->first->pt[temp->second].first,temp->first->pt[temp->second].second);
	FILE *out_file = fopen(file,"w");
	
	sprintf(file,"/mnt/data/recsys_spotify/line_data/ans_mean_%d_[%lld_%lld].out",temp->first->size,temp->first->pt[temp->second].first,temp->first->pt[temp->second].second);
	FILE *out_file_mean = fopen(file,"w");

	priority_queue< temp_prior >tt,tt1;
	vector< temp_prior > rever;
	vector<double> mean_vec(temp->first->size,0);


	for(i=temp->first->pt[temp->second].first ; i<temp->first->pt[temp->second].second ; i++)
	{
		printf("i:%lld\n",i);
		fflush(stdout);

		while(!tt.empty())
			tt.pop();
		
		while(!tt1.empty())
			tt1.pop();
		
		fill(mean_vec.begin(),mean_vec.end(),0);
		for(l=0;l<temp->first->query_in[i].size();l++)
			for(k=0 ; k< temp->first->size ; k++)
				mean_vec[k] += temp->first->arr[temp->first->query_in[i][l]][k];
		for(k=0 ; k< temp->first->size ; k++)
			mean_vec[k] /= temp->first->query_in[i].size();


		for(j=0 ; j<temp->first->arr.size() ; j++)
		{
			url = temp->first->query_url(j);
			if(url.find("spotify:track:") == std::string::npos)	//this is the title
				continue;

			all=0;
			for(l=0;l<temp->first->query_in[i].size();l++)
			{
				one = 0;
				for(k=0 ; k< temp->first->size ; k++)
				{
					one += (temp->first->arr[j][k] - temp->first->arr[temp->first->query_in[i][l]][k]) * (temp->first->arr[j][k] - temp->first->arr[temp->first->query_in[i][l]][k]);
				}
			}
			all += sqrt(one);
			
			h.node = j;
			h.value = all;
			tt.push(h);
			if(tt.size()>500)
				tt.pop();

			one = 0;
			if(temp->first->query_in[i].size()>0)
				for(k=0 ; k< temp->first->size ; k++)
					one += (mean_vec[k] - temp->first->arr[j][k])*(mean_vec[k] - temp->first->arr[j][k]);

			h.node = j;
			h.value = one;
			tt1.push(h);
			if(tt1.size()>500)
				tt1.pop();
		}
		printf("test%d %d\n",i,tt.size());	
		rever.clear();
		for( j=0 ; j<500 ; j++)
		{
			h = tt.top();
			tt.pop();
			rever.push_back(h);
		}
		for( j=499 ; j>=0 ; j--)
			fprintf(out_file,"%lld ",rever[j].node);
		fprintf(out_file,"\n");
		
		printf("test%d %d\n",i,tt1.size());	
		
		rever.clear();
		for( j=0 ; j<500 ; j++)
		{
			h = tt1.top();
			tt1.pop();
			rever.push_back(h);
		}
		for( j=499 ; j>=0 ; j--)
			fprintf(out_file_mean,"%lld ",rever[j].node);
		fprintf(out_file_mean,"\n");

	}
	fclose(out_file);
	fclose(out_file_mean);
	return 0;
}
void mapping::checking(int num,char* test_file,char* ans_file)
{
	int ss=num/this->thread_num,rc;
	
	ifstream in(test_file); 
	string temp;
	vector<string> quer[this->thread_num];
	for(int i=0;i<num;i++)
	{
		getline(in,temp);
		quer[i/ss].push_back(temp);
	}
	in.close();

	in.open(ans_file);
	vector<string> answ[this->thread_num];
	for(int i=0;i<num;i++)
	{
		getline(in,temp);	
		answ[i/ss].push_back(temp);
	}

	pthread_t thread[this->thread_num];
	temp_type parm[this->thread_num];
	void *res;

	printf("starting to make thread\n");

	for(int i=0;i<this->thread_num;i++)
	{
		thread[i] = i;

		parm[i].a = &quer[i];
		parm[i].b = &answ[i];
		parm[i].index = i;
		parm[i].re = this;

		rc = pthread_create(&thread[i],NULL,mapping::che,(void*)&parm[i]);
	}
	for(int i=0;i<this->thread_num;i++)
		rc = pthread_join(thread[i],&res);
}

void* mapping::che(void *parm)
{
	temp_type *temp = (temp_type*) parm;

	vector<string> *quer = temp->a;
	vector<string> *answ = temp->b;

	mapping* re = temp->re;

	int num,zero_count,number;
	size_t sz,en;
	double normal=0,pre;
	double correct_ndcg,correct_r;
	double correct_ndcg_max,correct_r_max;
	double correct_ndcg_min,correct_r_min;
	double correct_ndcg_avg,correct_r_avg;


	char file[128]="ans";
	sprintf(file,"./error_mean_%d.out",temp->index);
	
	ofstream ofs(file);
	
	string url;
	vector<int> test;

	bool key,on;
	int cc,song_int;
	
	correct_r_max = correct_ndcg_max = 0;
	correct_r_min = correct_ndcg_min = 0;
	correct_r_avg = correct_ndcg_avg = 0;

	string arr;
	zero_count = 0;
	for(long long i=0;i<quer->size();i++)
	{
		test.clear();

		number = stoi(answ->at(i),nullptr);

		sz = answ->at(i).find(" ");
		for(int j=0;j<number;j++)
		{
			en = answ->at(i).substr(sz+1).find(" ");
			song_int = re->query_int(answ->at(i).substr(sz+1,en));
			sz += en+1;

			if(song_int>0)
			{
				test.push_back(song_int);
				cout<<"test:"<<song_int<<endl;
			}
			else
				continue;
		}

		sz = en = 0;
		correct_r = correct_ndcg = normal = 0;

		if(number==0)
		{
			ofs<<correct_r<<" "<<correct_ndcg<<endl;
			continue;
		}
		arr = quer->at(i);
		
		on = false;
		
		cc=0;
		for(int j=0;j<500;j++)
		{
			key  = false;
			num = stoi(arr.substr(sz),&en);
			sz += en+1;
			
			//	cout<<"j:"<<pre<<endl;
		
			for(long long k=0 ; k<test.size() ; k++)
				if(num == test[k])
				{
					key = true;
					break;
				}
				
			if(key)
			{
				if(cc==0)
					normal +=1;
				else if(cc<number )
					normal += 1/log(cc+1);
				cc++;
				
				on = true;
				if(j<number)
					correct_r += 1;
				
				if(j==0)
					correct_ndcg += 1;
				else
					correct_ndcg += 1/log(j+1);
			}
		}
		if(on==false)
			normal += 1;

		correct_r /= number;
		correct_ndcg /= normal;
	
		ofs<<correct_r<<" "<<correct_ndcg<<endl;

		if(i==0)
		{
			correct_r_max = correct_r_min = correct_r;
			correct_ndcg_max = correct_ndcg_min = correct_ndcg;
		}
		else
		{
			if(correct_r_max < correct_r)
				correct_r_max = correct_r;
			if(correct_ndcg_max < correct_ndcg)
				correct_ndcg_max = correct_ndcg;
			
			if(correct_r_min > correct_r)
				correct_r_min = correct_r;
			if(correct_ndcg_min > correct_ndcg)
				correct_ndcg_min = correct_ndcg;
		}
	
		correct_r_avg += correct_r;
		correct_ndcg_avg += correct_ndcg;

		if(on ==false)
			zero_count++;
	}

	correct_r_avg /= quer->size();
	correct_ndcg_avg /= quer->size();
	ofs<<"---------------------------------------------\n";
	cout<<"---------------------------------------------\n";
	ofs<<"max:"<<temp->index<<" "<<correct_r_max<<" "<<correct_ndcg_max<<endl;
	//cout<<"max:"<<correct_r_max<<" "<<correct_ndcg_max<<endl;
	ofs<<"min:"<<correct_r_min<<" "<<correct_ndcg_min<<endl;
	//cout<<"min:"<<correct_r_min<<" "<<correct_ndcg_min<<endl;
	ofs<<"avg:"<<correct_r_avg<<" "<<correct_ndcg_avg<<endl;
	ofs<<"zero"<<zero_count<<endl;
	//cout<<"avg:"<<correct_r_avg<<" "<<correct_ndcg_avg<<endl;
	ofs.close();
	
	return 0; 
}
