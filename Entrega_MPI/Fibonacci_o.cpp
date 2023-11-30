#include <iostream>
#include <mpi.h>
using namespace std;
int fibonacci(int n) {
 if (n <= 1) {
 return n;
 }
 return fibonacci(n - 1) + fibonacci(n - 2);
}
int main(int argc, char** argv) {
 int rank, size;
 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 MPI_Comm_size(MPI_COMM_WORLD, &size);

 int n;
 if (rank == 0) {
 cout << "Ingrese la cantidad de elementos de la serie de Fibonacci: 
";
 cin >> n;
 }
 MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
 int a = 0, b = 1;
 int sum = 0;
 if (rank != 0) {
 MPI_Send(&sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
 }
 else {
 for (int i = 1; i < size; i++) {
 int temp;
 MPI_Recv(&temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD,
 MPI_STATUS_IGNORE);
 //sum += temp;
 }
 }
 MPI_Barrier(MPI_COMM_WORLD);
 if (rank == 0) {
 cout << "La serie Fibonacci es: ";
 for (int i = 0; i < n; i++) {
 int temp = a;
 a = b;
 b = temp + b;
 sum += temp;
 cout << temp << " ";
 }
 cout << endl;
 cout << "La suma de la serie Fibonacci es: " << sum << endl;
 }
 MPI_Finalize();
}