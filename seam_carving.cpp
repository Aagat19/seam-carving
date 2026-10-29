#include<iostream>
#include<cmath>
#include<string>
#include<vector>
#include<opencv2/opencv.hpp>
#include<csignal>
using namespace cv;
using namespace std;

void prog(int done,int total){
    double prog = (double)done/total*100.0 ;
    printf("\rProgress : %.2f%%",prog);
    fflush(stdout);
}

void handle_sigint(int){
    cerr<<"\nOperation cancelled by user."<<endl;
    exit(1);
}

// Dual-gradient energy: squared RGB differences of the left/right and up/down neighbours
double energyAtPoint(const Mat &img,int y,int x) {
    int ht=img.rows;
    int wid=img.cols;

    int left=max(x-1,0);
    int right=min(x+1,wid-1);
    int up=max(y-1,0);
    int down=min(y+1,ht-1);

    Vec3b pixLeft=img.at<Vec3b>(y,left);
    Vec3b pixRight=img.at<Vec3b>(y,right);
    Vec3b pixUp=img.at<Vec3b>(up,x);
    Vec3b pixDown=img.at<Vec3b>(down,x);

    int dxR=pixRight[2]-pixLeft[2];
    int dxG=pixRight[1]-pixLeft[1];
    int dxB=pixRight[0]-pixLeft[0];

    int dyR=pixDown[2]-pixUp[2];
    int dyG=pixDown[1]-pixUp[1];
    int dyB=pixDown[0]-pixUp[0];

    return sqrt(dxR*dxR+dxG*dxG+dxB*dxB+dyR*dyR+dyG*dyG+dyB*dyB);
}

Mat computeEnergyMatrix(const Mat& img) {
    int height=img.rows;
    int width=img.cols;
    Mat energy(height,width,CV_64F);

    for (int y=0;y<height;y++){
        for (int x=0;x<width;x++){
            energy.at<double>(y,x)=energyAtPoint(img,y,x);
        }
    }
    return energy;
}

void verticalSeamRemoval(Mat &img,Mat &energy){
    int row=img.rows;
    int col=img.cols;

    // M = cumulative minimum energy, P = direction (-1/0/+1) of the parent in the row above
    Mat M(row,col,CV_64F);
    Mat P(row,col,CV_32S);

    for(int j=0;j<col;j++){
        M.at<double>(0,j)=energy.at<double>(0,j);
        P.at<int>(0,j)=0;
    }

    for(int i=1;i<row;i++){
        for(int j=0;j<col;j++){
            double left=(j>0)? M.at<double>(i-1,j-1):1e9;
            double up=M.at<double>(i-1,j);
            double right=(j<col-1)?M.at<double>(i-1,j+1):1e9;

            double minimum_val=up;
            int flag=0;
            if(left<minimum_val){
                minimum_val=left;
                flag=-1;
            }
            if(right<minimum_val){
                minimum_val=right;
                flag=1;
            }

            M.at<double>(i,j)=energy.at<double>(i,j)+minimum_val;
            P.at<int>(i,j)=flag;
        }
    }

    vector<int> seam(row);
    int j_min=0;
    double minimum_val=M.at<double>(row-1,0);
    for(int j=1;j<col;j++){
        if(M.at<double>(row-1,j)<minimum_val){
            minimum_val=M.at<double>(row-1,j);
            j_min=j;
        }
    }
    seam[row-1] = j_min;

    // Backtrack from the bottom row using the parent directions
    for(int i=row-1;i>0;i--){
        int previous=seam[i]+P.at<int>(i,seam[i]);
        if(previous<0){
            previous=0;
        }
        if(previous>=col){
            previous=col-1;
        }
        seam[i-1]=previous;
    }

    for(int i=0;i<row;i++){
        for(int j=seam[i];j<col-1;j++){
            img.at<Vec3b>(i,j)=img.at<Vec3b>(i,j+1);
            energy.at<double>(i,j)=energy.at<double>(i,j+1);
        }
    }
    img=img.colRange(0,col-1);
    energy=energy.colRange(0,col-1);

    // Only pixels next to the removed seam have new neighbours, so only they need new energy
    for(int i=0;i<row;i++){
        int start=max(seam[i]-1,0);
        int end=min(seam[i]+1,img.cols-1);
        for(int j=start;j<=end;j++){
            energy.at<double>(i,j)=energyAtPoint(img,i,j);
        }
    }
}

void horizontalSeamRemoval(Mat &img,Mat &energy){
    transpose(img,img);
    transpose(energy,energy);
    verticalSeamRemoval(img,energy);
    transpose(img,img);
    transpose(energy,energy);
}

int main(){
    signal(SIGINT, handle_sigint);
    string path;
    cout<<"Enter image path: ";
    getline(cin,path);

    if(path.empty()){
        cerr<<"Error: Image path can't be empty!"<<endl;
        return -1;
    }

    Mat image=imread(path,IMREAD_COLOR);
    if(image.empty()){
        cerr<<"Error: Not able to find the image!"<<endl;
        return -1;
    }
    if(image.rows<=1 || image.cols<=1){
        cerr<<"Error: Image is too small to remove more seams."<<endl;
        return -1;
    }

    Mat energy=computeEnergyMatrix(image);

    int newRows,newCols;
    cout<<"Enter new height and new width: ";
    if(!(cin>>newRows>>newCols)){
        cerr<<"Error: Dimensions must be integers."<<endl;
        return -1;
    }

    if(newRows <= 0 || newCols <= 0){
        cerr<<"Error: Dimensions must be positive integers."<<endl;
        return -1;
    }
    if(newRows>image.rows || newCols>image.cols){
        cerr << "Error: New size must be smaller than or equal to original size."<<endl;
        return -1;
    }
    if(newRows==image.rows && newCols == image.cols) {
        cout<<"Note: New dimensions are same as original. No resizing needed."<<endl;
        return 0;
    }

    int verticalSeams=image.cols-newCols;
    int horizontalSeams=image.rows-newRows;

    for(int i=0;i<verticalSeams;i++){
        verticalSeamRemoval(image, energy);
        prog(i+1,verticalSeams+horizontalSeams);
    }

    for(int i=0;i<horizontalSeams;i++){
        horizontalSeamRemoval(image,energy);
        prog(verticalSeams+i+1,verticalSeams+horizontalSeams);
    }
    cout<<"\nDone!"<<endl;

    string outPath="output.jpg";
    if(!imwrite(outPath,image)){
        cerr<<"Error: Could not save the resized image."<<endl;
        return -1;
    }
    cout<<"Saved resized image ("<<image.cols<<"x"<<image.rows<<") to "<<outPath<<endl;

    imshow("Resized Image",image);
    waitKey(0);
    return 0;
}
