#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <typeinfo>
#include <cmath>
#include <algorithm>
#include <queue>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
namespace py = pybind11;


// define the object of each pixel position on range image
class PixelCoord {

public:
  int row=0;
  int col=0;
  PixelCoord(int row_, int col_) : row(row_), col(col_) {}

  PixelCoord operator+(const PixelCoord& other) const {
    return PixelCoord(row + other.row, col + other.col);
  }

};






class DM_Cluster{
    //default is private
    public:
    
    double angle_threshold=0.0; // angle threshold of two points
    double close_defined=3; 
    double ratio_threshold=0.0; // ration threshold to decide if merge
    double EC_dis_lower=0.5;
    double EC_dis_upper=1.0;


    
    static const int NEIGH_SIZE=4;
    static const int width=64;
    static const int height=2048;
    // this is how to construct the range image, from -3 degree to 25 degree in vertical direction, 64 is the image vertical resolution, 2048 is the horizontal resolution
    const double angle_resolution_x=28.0/64.0/180.0*3.14159;
    const double angle_resolution_y=3.14159*2/2048.0;
    // this define each single step to search the nearest point
    

    DM_Cluster(double input_thresh, double close_defined, double ratio_thresh, double EC_dis_lower_, double EC_dis_upper_):angle_threshold(input_thresh),close_defined(close_defined),ratio_threshold(ratio_thresh),EC_dis_lower(EC_dis_lower_),EC_dis_upper(EC_dis_upper_){}
    


    bool Condition_angle(PixelCoord &next_point, PixelCoord &current, double *range_img, double count_temp, double angle_resolution){
        auto d_1=std::max(range_img[next_point.row*this->height+next_point.col],range_img[current.row*this->height+current.col]);
        auto d_2=std::min(range_img[next_point.row*this->height+next_point.col],range_img[current.row*this->height+current.col]);
        auto current_indicator=atan(sin(count_temp*angle_resolution)*d_2/(d_1-d_2*cos(count_temp*angle_resolution)));
        //if (d_2>this->close_defined && count_temp>9) return false;
        //if  (count_temp>3) return false;
        if (d_2>1.5 && count_temp>11) return false;
        if (d_2>3 && count_temp>9) return false;
        if (d_2>5 && count_temp>7) return false;
        if (d_2>8 && count_temp>5) return false;
        if (d_2>15 && count_temp>3) return false;
        //if (d_2>30 && count_temp>3) return false;
        
        if (current_indicator>this->angle_threshold) return true;
        else return false;

        }

    double calculate_euclidean_distance(double *range_img, double *range_img_x, double *range_img_y, double *range_img_z, PixelCoord &current_point, PixelCoord &next_point){
        if (range_img[next_point.row*this->height+next_point.col]>200) return 1000.0;

        double result=std::pow(range_img_x[current_point.row*this->height+current_point.col]-range_img_x[next_point.row*this->height+next_point.col],2)+std::pow(range_img_y[current_point.row*this->height+current_point.col]-range_img_y[next_point.row*this->height+next_point.col],2)+std::pow(range_img_z[current_point.row*this->height+current_point.col]-range_img_z[next_point.row*this->height+next_point.col],2);
        return std::sqrt(result);
        }

    void Increase_queue(PixelCoord &current, double *range_img_x, double *range_img_y, double *range_img_z, double *range_img, std::array<int, 131072> &label_instance, std::queue<PixelCoord> &each_queue, std::vector<std::vector<int>> &merge_matrix_is_neighbor, std::vector<std::vector<int>> &merge_matrix_not_neighbor, std::array<int, 131072> &NN_location_left, std::array<int, 131072> &NN_location_right,std::array<int, 131072> &NN_location_up,std::array<int, 131072> &NN_location_down, std::vector<double> &x_max_matrix, std::vector<double> &x_min_matrix, std::vector<double> &y_max_matrix, std::vector<double> &y_min_matrix, std::vector<int> &count_matrix, std::vector<PixelCoord> &each_undecided, int current_label){
        label_instance[this->height*current.row+current.col]=current_label;
        double temp_x;
        double temp_y;

        
        if (NN_location_left[this->height*current.row+current.col]>-1){
            PixelCoord  move_left=PixelCoord(current.row,NN_location_left[this->height*current.row+current.col]);
            int count_temp=current.col-NN_location_left[this->height*current.row+current.col];
            bool if_same_label=Condition_angle(move_left,current, range_img, count_temp, this->angle_resolution_y);
            double temp_ec_dis=calculate_euclidean_distance(range_img,range_img_x,range_img_y,range_img_z,current,move_left);
            if_same_label=(if_same_label && temp_ec_dis<this->EC_dis_upper) || (temp_ec_dis<this->EC_dis_lower && range_img[this->height*current.row+current.col]<this->close_defined && range_img[this->height*move_left.row+move_left.col]<this->close_defined);
            //if_same_label=temp_ec_dis<0.5;
            if (if_same_label){
                if (label_instance[this->height*move_left.row+move_left.col]==0){
                    each_queue.push(move_left);
                    label_instance[this->height*move_left.row+move_left.col]=current_label;
                    temp_x=range_img_x[this->height*move_left.row+move_left.col];
                    temp_y=range_img_y[this->height*move_left.row+move_left.col];
                    if (temp_x>x_max_matrix[current_label-1]) x_max_matrix[current_label-1]=temp_x;
                    if (temp_x<x_min_matrix[current_label-1]) x_min_matrix[current_label-1]=temp_x;
                    if (temp_y>y_max_matrix[current_label-1]) y_max_matrix[current_label-1]=temp_y;
                    if (temp_y<y_min_matrix[current_label-1]) y_min_matrix[current_label-1]=temp_y; 
                    count_matrix[current_label-1]+=1;        
                    }else{
                    int next_label=label_instance[this->height*move_left.row+move_left.col];
                    merge_matrix_is_neighbor[next_label-1][current_label-1]+=1;
                    merge_matrix_is_neighbor[current_label-1][next_label-1]+=1;
                    }
                }else{
                    if (label_instance[this->height*move_left.row+move_left.col]==0){
                        each_undecided.push_back(move_left);
                    }else{
                        int next_label=label_instance[this->height*move_left.row+move_left.col];
                        merge_matrix_not_neighbor[next_label-1][current_label-1]+=1;
                        merge_matrix_not_neighbor[current_label-1][next_label-1]+=1;                       
                    }
                }
            }

        if (NN_location_right[this->height*current.row+current.col]>-1  ){
            PixelCoord  move_right=PixelCoord(current.row,NN_location_right[this->height*current.row+current.col]);
            int count_temp=NN_location_right[this->height*current.row+current.col]-current.col;


            bool if_same_label=Condition_angle(move_right,current, range_img, count_temp, this->angle_resolution_y);
            double temp_ec_dis=calculate_euclidean_distance(range_img,range_img_x,range_img_y,range_img_z,current,move_right);
            if_same_label=(if_same_label && temp_ec_dis<this->EC_dis_upper) || (temp_ec_dis<this->EC_dis_lower && range_img[this->height*current.row+current.col]<this->close_defined && range_img[this->height*move_right.row+move_right.col]<this->close_defined);
            //if_same_label=temp_ec_dis<0.5;
            if (if_same_label){
                if (label_instance[this->height*move_right.row+move_right.col]==0){
                    each_queue.push(move_right);
                    label_instance[this->height*move_right.row+move_right.col]=current_label;
                    temp_x=range_img_x[this->height*move_right.row+move_right.col];
                    temp_y=range_img_y[this->height*move_right.row+move_right.col];
                    if (temp_x>x_max_matrix[current_label-1]) x_max_matrix[current_label-1]=temp_x;
                    if (temp_x<x_min_matrix[current_label-1]) x_min_matrix[current_label-1]=temp_x;
                    if (temp_y>y_max_matrix[current_label-1]) y_max_matrix[current_label-1]=temp_y;
                    if (temp_y<y_min_matrix[current_label-1]) y_min_matrix[current_label-1]=temp_y;
                    count_matrix[current_label-1]+=1;     
                    }else{
                    int next_label=label_instance[this->height*move_right.row+move_right.col];
                    merge_matrix_is_neighbor[next_label-1][current_label-1]+=1;
                    merge_matrix_is_neighbor[current_label-1][next_label-1]+=1;
                    }
                }else{
                    if (label_instance[this->height*move_right.row+move_right.col]==0){
                        each_undecided.push_back(move_right);
                    }else{
                        int next_label=label_instance[this->height*move_right.row+move_right.col];
                        merge_matrix_not_neighbor[next_label-1][current_label-1]+=1;
                        merge_matrix_not_neighbor[current_label-1][next_label-1]+=1;                       
                    }
                }
            }

        if (NN_location_up[this->height*current.row+current.col]>-1){
            PixelCoord  move_up=PixelCoord(NN_location_up[this->height*current.row+current.col],current.col);
            int count_temp=NN_location_up[this->height*current.row+current.col]-current.row;
            bool if_same_label=Condition_angle(move_up,current, range_img, count_temp, this->angle_resolution_x);
            double temp_ec_dis=calculate_euclidean_distance(range_img,range_img_x,range_img_y,range_img_z,current,move_up);
            if_same_label=(if_same_label && temp_ec_dis<this->EC_dis_upper) || (temp_ec_dis<this->EC_dis_lower && range_img[this->height*current.row+current.col]<this->close_defined && range_img[this->height*move_up.row+move_up.col]<this->close_defined);
            //if_same_label=temp_ec_dis<0.5;
            if (if_same_label){
                if (label_instance[this->height*move_up.row+move_up.col]==0){
                    each_queue.push(move_up);
                    label_instance[this->height*move_up.row+move_up.col]=current_label;
                    temp_x=range_img_x[this->height*move_up.row+move_up.col];
                    temp_y=range_img_y[this->height*move_up.row+move_up.col];
                    if (temp_x>x_max_matrix[current_label-1]) x_max_matrix[current_label-1]=temp_x;
                    if (temp_x<x_min_matrix[current_label-1]) x_min_matrix[current_label-1]=temp_x;
                    if (temp_y>y_max_matrix[current_label-1]) y_max_matrix[current_label-1]=temp_y;
                    if (temp_y<y_min_matrix[current_label-1]) y_min_matrix[current_label-1]=temp_y;
                    count_matrix[current_label-1]+=1;
                    }else{
                    int next_label=label_instance[this->height*move_up.row+move_up.col];
                    merge_matrix_is_neighbor[next_label-1][current_label-1]+=1;
                    merge_matrix_is_neighbor[current_label-1][next_label-1]+=1;
                    }
                }else{
                    if (label_instance[this->height*move_up.row+move_up.col]==0){
                        each_undecided.push_back(move_up);
                    }else{
                        int next_label=label_instance[this->height*move_up.row+move_up.col];
                        merge_matrix_not_neighbor[next_label-1][current_label-1]+=1;
                        merge_matrix_not_neighbor[current_label-1][next_label-1]+=1;                       
                    }
                }
            }

        if (NN_location_down[this->height*current.row+current.col]>-1){
            PixelCoord  move_down=PixelCoord(NN_location_down[this->height*current.row+current.col],current.col);
            int count_temp=current.row-NN_location_down[this->height*current.row+current.col];
            bool if_same_label=Condition_angle(move_down,current, range_img, count_temp, this->angle_resolution_x);
            double temp_ec_dis=calculate_euclidean_distance(range_img,range_img_x,range_img_y,range_img_z,current,move_down);
            if_same_label=(if_same_label && temp_ec_dis<this->EC_dis_upper) || (temp_ec_dis<this->EC_dis_lower && range_img[this->height*current.row+current.col]<this->close_defined && range_img[this->height*move_down.row+move_down.col]<this->close_defined);
            //if_same_label=temp_ec_dis<0.5;
            if (if_same_label){
                if (label_instance[this->height*move_down.row+move_down.col]==0){
                    each_queue.push(move_down);
                    label_instance[this->height*move_down.row+move_down.col]=current_label;
                    temp_x=range_img_x[this->height*move_down.row+move_down.col];
                    temp_y=range_img_y[this->height*move_down.row+move_down.col];
                    if (temp_x>x_max_matrix[current_label-1]) x_max_matrix[current_label-1]=temp_x;
                    if (temp_x<x_min_matrix[current_label-1]) x_min_matrix[current_label-1]=temp_x;
                    if (temp_y>y_max_matrix[current_label-1]) y_max_matrix[current_label-1]=temp_y;
                    if (temp_y<y_min_matrix[current_label-1]) y_min_matrix[current_label-1]=temp_y;
                    count_matrix[current_label-1]+=1;
                    }else{
                    int next_label=label_instance[this->height*move_down.row+move_down.col];
                    merge_matrix_is_neighbor[next_label-1][current_label-1]+=1;
                    merge_matrix_is_neighbor[current_label-1][next_label-1]+=1;
                    }
                }else{
                    if (label_instance[this->height*move_down.row+move_down.col]==0){
                        each_undecided.push_back(move_down);
                    }else{
                        int next_label=label_instance[this->height*move_down.row+move_down.col];
                        merge_matrix_not_neighbor[next_label-1][current_label-1]+=1;
                        merge_matrix_not_neighbor[current_label-1][next_label-1]+=1;                       
                    }
                }
            }
        }


    std::array<int, 131072>  DM_cluster(py::array_t<double> input_array_x_,py::array_t<double> input_array_y_, py::array_t<double> input_array_z_, py::array_t<double> input_array,py::array_t<int> mm_seed,py::array_t<int> nn_seed, int num_seed){
    	    
        // define the label output, initialize with all zeros
        std::array<int, 131072> label_instance=std::array<int, 131072>();
        for (int i=0;i<131072;i++) label_instance[i]=0;
        
        // use np.reshape(-1) to strip the 2D image to 1d array
        auto input_array_x = input_array_x_.request();
        double *ptr_x = (double *) input_array_x.ptr;

        auto input_array_y = input_array_y_.request();
        double *ptr_y = (double *) input_array_y.ptr;

        auto input_array_z = input_array_z_.request();
        double *ptr_z = (double *) input_array_z.ptr;

		auto buf1 = input_array.request();
		double *ptr1 = (double *) buf1.ptr;

        auto mm_ptr = mm_seed.request();
        int *mm_ptr1 = (int *) mm_ptr.ptr;
       
        auto nn_ptr = nn_seed.request();
        int *nn_ptr1 = (int *) nn_ptr.ptr;

        //define NN position
        std::array<int, 131072> NN_location_left=std::array<int, 131072>();
        std::array<int, 131072> NN_location_right=std::array<int, 131072>();
        std::array<int, 131072> NN_location_up=std::array<int, 131072>();
        std::array<int, 131072> NN_location_down=std::array<int, 131072>();
        
        int look_up=0;
        int look_down=0;

        for (int i=0; i<this->width;i++){
            int the_left=-1;
            for (int j=0; j<this->height;j++){
                if (ptr1[i*this->height+j]>0.01){
                    if (the_left>-1){
                        NN_location_right[i*this->height+the_left]=j;    
                        }
                    NN_location_left[i*this->height+j]=the_left;
                    the_left=j;
                    while(look_up<10){
                        look_up+=1;
                        if (i+look_up+1>this->width){
                            NN_location_up[i*this->height+j]=-1;
                            break;
                        }
                        if (ptr1[(look_up+i)*this->height+j]>0.01){
                            NN_location_up[i*this->height+j]=i+look_up;
                            break;
                        }
                    }
                    look_up=0;

                    while(look_down<10){
                        look_down+=1;
                        if (i-look_up<0){
                            NN_location_down[i*this->height+j]=-1;
                            break;
                        }
                        if (ptr1[(i-look_down)*this->height+j]>0.01){
                            NN_location_down[i*this->height+j]=i-look_down;
                            break;
                        }
                    }
                    look_down=0;


                    }else{
                        NN_location_left[i*this->height+j]=-1;
                        NN_location_right[i*this->height+j]=-1;
                        NN_location_up[i*this->height+j]=-1;
                        NN_location_down[i*this->height+j]=-1;
                    }

                }
            NN_location_right[i*this->height+the_left]=-1;
            }
        


        // define the queue vector
        std::vector<std::queue<PixelCoord>> vec_queue; // vector of queues
        for (size_t idx = 0; idx < num_seed; idx++){
            vec_queue.push_back(std::queue<PixelCoord>());
            vec_queue[idx].push(PixelCoord(mm_ptr1[idx],nn_ptr1[idx]));
            }

        // define vector to save undecided neighbor
        std::vector<std::vector<PixelCoord>> undecide_neighbor; // vector of vector to store un-decided neighbor
        for (int i=0;i<num_seed;i++){
            std::vector<PixelCoord> temp_vector;
            undecide_neighbor.push_back(temp_vector);
            }

    

        
        // define the merging matrix
        std::vector<std::vector<int>> merge_matrix_is_neighbor;
        for (int i=0;i<num_seed;i++){
            std::vector<int> temp_vector;
            for (int j=0;j<num_seed;j++){
                temp_vector.push_back(0);
                }
            merge_matrix_is_neighbor.push_back(temp_vector);
            }


        std::vector<std::vector<int>> merge_matrix_not_neighbor;
        for (int i=0;i<num_seed;i++){
            std::vector<int> temp_vector;
            for (int j=0;j<num_seed;j++){
                temp_vector.push_back(0);
                }
            merge_matrix_not_neighbor.push_back(temp_vector);
            }


        // define the corner matrix
        std::vector<double> x_max_matrix;
        std::vector<double> x_min_matrix;
        std::vector<double> y_max_matrix;
        std::vector<double> y_min_matrix;
        std::vector<int> count_matrix;
        for (int i=0;i<num_seed;i++){
            x_max_matrix.push_back(-10000);
            y_max_matrix.push_back(-10000);
            x_min_matrix.push_back(10000);
            y_min_matrix.push_back(10000);
            count_matrix.push_back(0);
            }

    




        // multi-local search
        int total_empty_queue_count=0;
        while (total_empty_queue_count<num_seed){
            total_empty_queue_count=0;
            for (int i =0; i< num_seed; i++){
                if (vec_queue[i].size()==0) {
                    total_empty_queue_count+=1;
                }
                else{
                    PixelCoord current = vec_queue[i].front();
                    vec_queue[i].pop();
                    Increase_queue(current, ptr_x ,ptr_y, ptr_z, ptr1,label_instance, vec_queue[i], merge_matrix_is_neighbor,  merge_matrix_not_neighbor, NN_location_left, NN_location_right,NN_location_up,NN_location_down, x_max_matrix, x_min_matrix, y_max_matrix, y_min_matrix, count_matrix, undecide_neighbor[i], i+1);
                    }
                }
            }

        

        // add undecided neighbors into merge_matrix
        for (int i=0;i<num_seed;i++){
            auto each_vector=undecide_neighbor[i];
            if (each_vector.size()==0) continue;
            for (int j=0;j<each_vector.size();j++){
                auto other_label=label_instance[this->height*each_vector[j].row+each_vector[j].col];
                if (other_label>0){
                    merge_matrix_not_neighbor[other_label-1][i]+=1;
                    merge_matrix_not_neighbor[i][other_label-1]+=1;
                    }
                }
            }
        

        
        //define the hash list
        std::vector<int> hash_merged;
        for (int i=0;i<num_seed;i++) hash_merged.push_back(0);
        std::vector<int> wait_assign;
        for (int i=0;i<num_seed;i++) wait_assign.push_back(i);

        int indi_label=1;
        std::vector<double> indi_x_max;
        std::vector<double> indi_x_min;
        std::vector<double> indi_y_max;
        std::vector<double> indi_y_min;
        std::vector<int> indi_count;


        while (wait_assign.size()>0){
            int current_index=wait_assign[0];
            hash_merged[wait_assign[0]]=indi_label;
            indi_x_max.push_back(x_max_matrix[current_index]);
            indi_x_min.push_back(x_min_matrix[current_index]);
            indi_y_max.push_back(y_max_matrix[current_index]);
            indi_y_min.push_back(y_min_matrix[current_index]);
            indi_count.push_back(count_matrix[current_index]);
            wait_assign.erase(wait_assign.begin());
            std::queue<int> label_queue;
            int j=0;
            while (j<wait_assign.size()){
                int if_add=1;
                float temp_a=merge_matrix_is_neighbor[current_index][wait_assign[j]];
                float temp_b=merge_matrix_not_neighbor[current_index][wait_assign[j]];
                auto ratio_threshold=temp_a/(temp_b+temp_a);

                if (ratio_threshold>this->ratio_threshold){
                    hash_merged[wait_assign[j]]=indi_label;
                    for (int zz=0;zz<num_seed;zz++){
                        merge_matrix_is_neighbor[current_index][zz]=merge_matrix_is_neighbor[current_index][zz]+merge_matrix_is_neighbor[wait_assign[j]][zz];
                        merge_matrix_not_neighbor[current_index][zz]=merge_matrix_not_neighbor[current_index][zz]+merge_matrix_not_neighbor[wait_assign[j]][zz];
                    }
                    if (indi_x_max[indi_label-1]<x_max_matrix[wait_assign[j]]) indi_x_max[indi_label-1]=x_max_matrix[wait_assign[j]];
                    if (indi_x_min[indi_label-1]>x_min_matrix[wait_assign[j]]) indi_x_min[indi_label-1]=x_min_matrix[wait_assign[j]];
                    if (indi_y_max[indi_label-1]<y_max_matrix[wait_assign[j]]) indi_y_max[indi_label-1]=y_max_matrix[wait_assign[j]];
                    if (indi_y_min[indi_label-1]>y_min_matrix[wait_assign[j]]) indi_y_min[indi_label-1]=y_min_matrix[wait_assign[j]];
                    indi_count[indi_label-1]+=count_matrix[wait_assign[j]];
                    label_queue.push(wait_assign[j]);
                    wait_assign.erase(wait_assign.begin()+j);
                    if_add=0;
                    }
                if (if_add==1) j+=1;
                }

      
            while (!label_queue.empty()) {
                int each_index=label_queue.front();
                label_queue.pop();
                int j=0;
                while (j<wait_assign.size()){ 
                    int if_add=1;
                    float temp_a=merge_matrix_is_neighbor[each_index][wait_assign[j]];
                    float temp_b=merge_matrix_not_neighbor[each_index][wait_assign[j]];
                    auto ratio_threshold=temp_a/(temp_b+temp_a);
                    
                    if (ratio_threshold>this->ratio_threshold){
                        hash_merged[wait_assign[j]]=indi_label;
                        for (int zz=0;zz<num_seed;zz++){
                            merge_matrix_is_neighbor[current_index][zz]=merge_matrix_is_neighbor[current_index][zz]+merge_matrix_is_neighbor[wait_assign[j]][zz];
                            merge_matrix_not_neighbor[current_index][zz]=merge_matrix_not_neighbor[current_index][zz]+merge_matrix_not_neighbor[wait_assign[j]][zz];
                        }

                        if (indi_x_max[indi_label-1]<x_max_matrix[wait_assign[j]]) indi_x_max[indi_label-1]=x_max_matrix[wait_assign[j]];
                        if (indi_x_min[indi_label-1]>x_min_matrix[wait_assign[j]]) indi_x_min[indi_label-1]=x_min_matrix[wait_assign[j]];
                        if (indi_y_max[indi_label-1]<y_max_matrix[wait_assign[j]]) indi_y_max[indi_label-1]=y_max_matrix[wait_assign[j]];
                        if (indi_y_min[indi_label-1]>y_min_matrix[wait_assign[j]]) indi_y_min[indi_label-1]=y_min_matrix[wait_assign[j]];
                        indi_count[indi_label-1]+=count_matrix[wait_assign[j]];
                        label_queue.push(wait_assign[j]);
                        wait_assign.erase(wait_assign.begin()+j);
                        if_add=0;

                        }
                    if (if_add==1) j+=1;
                    
                        
                    }

                }
            
            indi_label+=1;
            } 



        indi_label=indi_label-1;
        std::vector<int> final_label;
        // final_label[hash_merge[label-1]-1]
        for (int i=0; i<indi_label;i++){
             final_label.push_back(i+1);
            }

        
        for (int i=0; i<indi_label-1;i++){
            int current_label=final_label[i];
            double current_x_max=indi_x_max[i];
            double current_x_min=indi_x_min[i];
            double current_y_max=indi_y_max[i];
            double current_y_min=indi_y_min[i];
            for (int j=i+1; j<indi_label;j++){
                double prob_x_max=indi_x_max[j];
                double prob_x_min=indi_x_min[j];
                double prob_y_max=indi_y_max[j];
                double prob_y_min=indi_y_min[j];

                double common_x_max=std::min(current_x_max,prob_x_max);
                double common_x_min=std::max(current_x_min,prob_x_min);
                double common_y_max=std::min(current_y_max,prob_y_max);
                double common_y_min=std::max(current_y_min,prob_y_min);
                
                bool overlap_condition_1=((common_x_max+0.1>common_x_min) && (common_y_max+0.1>common_y_min));// || ((common_x_max>common_x_min) && (common_y_max+0.1>common_y_min));
                
                bool overlap_condition=overlap_condition_1;
                if (overlap_condition){
                    final_label[j]=current_label;
                    indi_x_max[j]=std::max(prob_x_max,current_x_max);
                    indi_x_min[j]=std::min(prob_x_min,current_x_min);
                    indi_y_max[j]=std::max(prob_y_max,current_y_max);
                    indi_y_min[j]=std::min(prob_y_min,current_y_min);
                    indi_count[j]+=indi_count[i];
                    }

                }
            }

        


        for (int idx = 0; idx < width; idx++){
            for (int idy = 0; idy < height; idy++){
                if (label_instance[this->height*idx+idy]>0){
                    label_instance[this->height*idx+idy]=final_label[hash_merged[label_instance[this->height*idx+idy]-1]-1];
                    }

                }
            }
        
		return label_instance;
        }
    };





PYBIND11_MODULE(DM_Cluster, m) {
    py::class_<DM_Cluster>(m, "DM_Cluster")
    	.def(py::init<float, float, float, float, float>())
        .def("Condition_angle", &DM_Cluster::Condition_angle)
        .def("Increase_queue", &DM_Cluster::Increase_queue)
        .def("calculate_eu_dis",&DM_Cluster::calculate_euclidean_distance)
        .def("DM_Cluster", &DM_Cluster::DM_cluster);
}

