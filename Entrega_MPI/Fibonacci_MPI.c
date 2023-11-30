#include <iostream>
#include <mpi.h>

using namespace std;

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int temp = a;
        a = b;
        b = temp + b;
    }
    return b;
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    if (rank == 0) {
        cout << "Ingrese la cantidad de elementos de la serie de Fibonacci: ";
        cin >> n;

        if (n < 0) {
            cerr << "Por favor, ingrese un número no negativo para n." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_sum = 0;
    for (int i = rank; i < n; i += size) {
        local_sum += fibonacci(i);
    }

    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "La suma de la serie Fibonacci es: " << global_sum << endl;
    }

    MPI_Finalize();
    return 0;
}
