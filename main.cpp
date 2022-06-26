#include <iostream>
#include <bits/stdc++.h>
#include <vector>
#include <fstream>
#include "omp.h"

using namespace std;


//IMPORTANT : please consider that there is a variable matrixSize in the
//main function you have to change it if you change the matrix.


//This function read's txt file and
//fill the matrix var with matrix.
int readMatrixFile(vector<vector<int>> &matrix, int N, char *fileName){
    ifstream file(fileName);
    if(file.is_open()){

        // reading matrix values
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                file >> matrix[i][j];
            }
        }

        // closing file
        file.close();

    } else {
        cout << "Error opening file: " << fileName << endl;
        return 1;
    }
    return 0;
}

//This function add 2 matrix A and B and put the result in C
//we parallel the for because there is no dependencies in each iteration
void add (vector<vector<int>> &A, vector<vector<int>> &B, vector<vector<int>> &C, int size){
    #pragma omp parallel for
    for(int i = 0 ; i <A.size() ; i++){
        for (int j = 0 ; j<A[0].size();j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

//This function sub 2 matrix A and B and put the result in C
//we parallel the for because there is no dependencies in each iteration
void sub(vector<vector<int>> &A, vector<vector<int>> &B, vector<vector<int>> &C, int size)
{
    #pragma omp parallel for
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[0].size(); j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

//This function multiply 2 matrix A and B and put the result in C
void multiply(vector<vector<int>> &A, vector<vector<int>> &B, vector<vector<int>> &C, int size)
{
    for(int i=0; i < A.size(); i++){
        for(int j=0; j < A[0].size(); j++){
            for(int k=0; k < A[0].size(); k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

//This is the strassen function
void strassen(vector<vector<int>> &A, vector<vector<int>> &B, vector<vector<int>> &C, int N){

    //This is the threshold of the matrix len and if the size of the rows is
    //less than 4 we multiply the matrix in the normal way.
    if (N <= 4){
        return multiply(A , B , C , N);
    }

    int new_size = N/2;
    vector<int> z(new_size);

    //allocating the data that we need
    vector<vector<int>>
            a11(new_size, z), a12(new_size, z), a21(new_size, z), a22(new_size, z),
            b11(new_size, z), b12(new_size, z), b21(new_size, z), b22(new_size, z),
            c11(new_size, z), c12(new_size, z), c21(new_size, z), c22(new_size, z),
            m1(new_size, z), m2(new_size, z), m3(new_size, z), m4(new_size, z),
            m5(new_size, z), m6(new_size, z), m7(new_size, z),
            aResult(new_size, z), bResult(new_size, z);


    //Split the A to a11 , a12 , a21 , a22
    //Split the B to b11 , b12 , b21 , b22
    int i,j;
    for(i = 0;i < new_size; ++i)
    {
        for(j = 0; j < new_size;++j){
            a11[i][j] = A[i][j];
            a12[i][j] = A[i][j + new_size];
            a21[i][j] = A[i + new_size][j];
            a22[i][j] = A[i + new_size][j + new_size];

            b11[i][j] = B[i][j];
            b12[i][j] = B[i][j + new_size];
            b21[i][j] = B[i + new_size][j];
            b22[i][j] = B[i + new_size][j + new_size];
        }
    }

#pragma omp parallel firstprivate(aResult,bResult)
    {
        #pragma omp single
        {
            //calculating M1 part of strassen
            //M1 = (a11 + a22)(b11 + b22)
            #pragma omp task
            {
                add(a11, a22, aResult, new_size);
                add(b11, b22, bResult, new_size);
                strassen(aResult,bResult,m1, new_size);
            }
            //calculating M2 part of the strassen
            //M2 = (a11 + a22)b11
            #pragma omp task
            {
                add(a21, a22,aResult, new_size);
                strassen(aResult,b11,m2, new_size);
            }
            //calculating M3 part of the strassen
            //M3 = a11(b12- b22)
            #pragma omp task
            {
                sub(b12, b22, bResult, new_size);
                strassen(a11, bResult, m3, new_size);
            }
            //calculating M4 part of the strassen
            //M4 = a22(b21 - b11)
            #pragma omp task
            {
                sub(b21, b11, bResult, new_size);
                strassen(a22, bResult, m4, new_size);
            }
            //calculating M5 part of the strassen
            //M5 = (a11 + a12)b22
            #pragma omp task
            {
                add(a11, a12, aResult, new_size);
                strassen(aResult, b22, m5, new_size);
            }
            //calculating M6 part of the strassen
            //M6 = (a21 - a11)(b11 + b12)
            #pragma omp task
            {
                sub(a21, a11, aResult, new_size);
                add(b11, b12, bResult, new_size);
                strassen(aResult, bResult, m6, new_size);
            }
            //calculating M7 part of the strassen
            //M7 = (a12 - a22)(b21 - b22)
            #pragma omp task
            {
                sub(a12, a22, aResult, new_size);
                add(b21, b22, bResult, new_size);
                strassen(aResult, bResult, m7, new_size);
            }

            //waiting all above 7 tasks to complete here.
            #pragma omp taskwait

            //calculate c12 (c is the result)
            //c12 = M3 + M5
            #pragma omp task
            {
                add(m3, m5, c12, new_size);
            }
            //calculating c21
            //c21 = M2 + M4
            #pragma omp task
            {
                add(m2, m4, c21, new_size);
            }
            //calculating the c11
            //c11 = M1 + M4 - M5 + M7
            #pragma omp task
            {
                add(m1, m4, aResult, new_size);
                add(aResult, m7, bResult, new_size);
                sub(bResult, m5, c11, new_size);
            }
            //calculating the c22
            //c22 = M1 - M2 + M3 + M6
            #pragma omp task
            {
                sub(m1, m2, aResult, new_size);
                add(aResult, m3, bResult, new_size);
                add(bResult, m6, c22, new_size);
            }

            //waiting the c11 , c12 , c21 , c22 calculate
            #pragma omp taskwait

            //copy the c11 , c12 , c21 , c22 to the C
            for(i = 0; i < new_size; ++i){
                for(j = 0; j < new_size; ++j){
                    C[i][j] = c11[i][j];
                    C[i][j + new_size] = c12[i][j];
                    C[i + new_size][j] = c21[i][j];
                    C[i + new_size][j + new_size] = c22[i][j];
                }
            }
        };
    }

}

int main() {
    std::cout << "Hello, World!" << std::endl;

    //size of matrix for example in code below it's 16*16
    int matrixSize = 8 ;

    vector<int> v(matrixSize);


    //init the matrix A
    vector<vector<int>> A(matrixSize , v);
    readMatrixFile(A , matrixSize, "/home/hossein/CLionProjects/sterassen/matrix1.txt");

    //init matrix B
    vector<vector<int>> B(matrixSize, v);
    readMatrixFile(B , matrixSize, "/home/hossein/CLionProjects/sterassen/matrix2.txt");

    vector<vector<int>> c (matrixSize , v);

    clock_t start = clock();

    strassen(A , B , c , matrixSize);

    clock_t end = clock();
    double time_taken = (double)(end- start)/CLOCKS_PER_SEC;
    cout<<time_taken<< setprecision(5) <<endl;

    for (int i = 0 ; i < matrixSize ; i++){
        for (int j = 0 ; j < matrixSize ; j++) {
            cout<<c[i][j]<<" ";
        }
        cout<<endl;
    }
    return 0;
}
