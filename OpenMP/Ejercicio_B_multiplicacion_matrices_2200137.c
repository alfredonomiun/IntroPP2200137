#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

int main() {
    printf("Iniciando ejecucion del Script...\n \n");
    int A = 70000; // Numero  filas matriz 1
    int B = 30000; // Numero de columnas  matriz 1 y Numero de filas matriz 2
    int C = 12300; // Numero de columnas matriz 2

    printf("Tamano de matrices de entrada:\n");
    printf("Matriz 1 = %d x %d\n", A, B);
    printf("Matriz 2 = %d x %d\n", B, C);

    // Declarar punteros a matrices
    int **matriz1;
    int **matriz2;
    int **respuesta;

    // Asignar memoria dinamica para las matrices
    matriz1 = (int **)malloc(A * sizeof(int *));
    matriz2 = (int **)malloc(B * sizeof(int *));
    respuesta = (int **)malloc(A * sizeof(int *));
    for (int i = 0; i < A; i++) {
        matriz1[i] = (int *)malloc(B * sizeof(int));
        respuesta[i] = (int *)malloc(C * sizeof(int));
    }
    for (int i = 0; i < B; i++) {
        matriz2[i] = (int *)malloc(C * sizeof(int));
    }

    // Inicializar la semilla para generar numeros aleatorios
    srand(time(NULL));

    // Inicializar las matrices con valores aleatorios entre 1 y 100
    printf("\n");
    printf("Inicializando matriz 1...\n");
    for (int i = 0; i < A; i++) {
        for (int j = 0; j < B; j++) {
            matriz1[i][j] = (rand() % 100) + 1;
        }
    }

    printf("\n");
    printf("Inicializando matriz 2...\n");
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < C; j++) {
            matriz2[i][j] = (rand() % 100) + 1;
            //printf("%d ", matriz2[i][j]);
        }
        //printf("\n");
    }

    // Inicializar la matriz de respuesta
    printf("\n Inicializando matriz del respuesta...\n");
    for (int i = 0; i < A; i++) {
        for (int j = 0; j < C; j++) {
            respuesta[i][j] = 0;

        }
        //printf("\n");
    }

    // Realizar la Multiplicacion de matrices
    printf("\n Realizando la multiplicacion de matrices (de manera secuencial)...\n");
    for (int i = 0; i < A; i++) {
        for (int j = 0; j < C; j++) {
            for (int k = 0; k < B; k++) {
                respuesta[i][j] += matriz1[i][k] * matriz2[k][j];
            }
        }
    }

    printf("\n Tamaño de la matriz de salida:\n");
    printf("Matriz de salida = %d x %d\n", A, C);

    // Imprimir la matriz respuesta
    //printf("\n respuesta de la multiplicacion de matrices:\n");
    for (int i = 0; i < A; i++) {
        for (int j = 0; j < C; j++) {
        }
    }

    // Liberar memoria
    printf("\n Liberando memoria...\n Finalizando ejecucion del script... \n");
    for (int i = 0; i < A; i++) {
        free(matriz1[i]);
        free(respuesta[i]);
    }
    for (int i = 0; i < B; i++) {
        free(matriz2[i]);
    }
    free(matriz1);
    free(matriz2);
    free(respuesta);

    return 0; // Salir con EXITO
}
