## Explicacion codigo orignial fibonacci_o

Lo primero que hago es entender la solucion encontrada, entonces, en este ejercicio lo primero que se hace es definir las bibliotecas iostream y mpi.h las cuales nos permiten utilizar las funciones y clases dadas por MPI

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/b94f36aa-205b-4fb2-ad20-7ca79f61d704)

siguiendo, tenemos una funcion diseñada para determinar el numero de Fibonacci correspondiente a un valor proporcionado, el cual sera ingresado mediante teclado en un momento posterior. La funcion incluye una condicion que establece que, si el valor de n es menor o igual a 1, se devuelve el propio valor de n, ya que 0 y 1 son los dos primeros numeros de la serie de Fibonacci. En el caso contrario, si n no es menor o igual a 1, la funcion se invoca a si misma dos veces para calcular los numeros  Fibonacci anteriores a traves de (n-1) y (-n-2).
Este enfoque se basa en la regla fundamental de la serie de Fibonacci, donde cada numero es la suma de sus dos predecesores. Por lo tanto, se calcula sumando los elementos Fibonacci(i-1) y Fibonacci(i-2)
![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/a6274fa0-702d-4c00-8ca4-a0506de90d8a)

Se implementa una función denominada main, la cual utiliza el estándar de la biblioteca de MPI (Message Passing Interface) con el propósito de facilitar la comunicación entre varios procesos. Esta función toma como argumentos argc y argv, lo que posibilita la interacción entre los distintos procesos. La utilización de MPI permite la transferencia de mensajes entre los procesos, contribuyendo así a la coordinación y colaboración eficiente entre ellos.

Para obtener el rango y el tamaño se utiliza las funciones de MPI_Comm_rank y 
MPI_Comm_size que nos ayudara a obtener el rango es decir el identificador y un 
MPI_COMM_WORLD el numero total de procesos del comunicador. Estos valores son 
almacenados en la variable Rank y size
![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/854538d2-15d8-4749-85c5-3786d4e25fa3)

Lo siguiente a realizar es declarar una variable n para almacenar la cantidad de elementos de la serie. Si el rango es 0, Se solicita al usuario ingresar la cantidad de elementos de la serie de Fibonacci. el valor que se ingrese se almacenara en n
![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/e589904e-d285-4b5f-accd-73e19c9c33bd)

Ahora, se utiliza la función MPI_Bcast 
para transmitir el valor de n desde el proceso 0 a todos los demás procesos en el 
comunicador MPI_COMM_WORLD
Se inician las variables a,b y sum. Las cuales se usaran para calcular la serie de Fibonacci y almacenar la correspondiente suma de los numeros,Se llama a la función MPI_Barrier para sincronizar todos los procesos

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/4515cb9a-c1ce-49bb-9208-24e3dde11475)



## MEJORAS HECHAS

entonces, tenemos el archivo Fibonacci_MPI el cual contiene las mejoras hechas por mi persona, las cuales son:

1- Evitar la recursión para Fibonacci:
En lugar de usar una función recursiva para calcular Fibonacci, se implementó una versión iterativa de la función fibonacci. La recursión podría ser ineficiente para valores grandes debido a la duplicación de cálculos.
![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/49eef840-cd97-4ee7-b1a8-adaf5f3ee796)

2- Manejar errores de entrada:
Se añadió una validación para asegurarse de que el usuario ingrese un valor válido para n. Si el valor ingresado es negativo, se imprime un mensaje de error y el programa se aborta con MPI_Abort.
![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/caa7254e-5e8a-46d6-8593-71ad77fc110c)

3-Mejorar la distribución del trabajo:
El bucle para calcular la serie Fibonacci se modificó para que cada proceso MPI calcule una parte de la serie de manera más equitativa, distribuyendo la carga de trabajo.
![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/29216482-9a16-4548-af29-f6b7647c34b4)

4- Usar MPI_Reduce para la suma global:
Se reemplazó el envío y recepción manual de resultados con la función MPI_Reduce para realizar la suma global de las sumas locales de cada proceso.
![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/fbfe0798-8b6b-4207-9e15-3037c80640ff)

## Explicacion porque da mejor tiempo


La mejora en el tiempo de ejecución al usar MPI (Message Passing Interface) en lugar de un enfoque secuencial se debe principalmente a la capacidad de MPI para distribuir la carga de trabajo entre varios procesos y aprovechar los recursos paralelos disponibles.

Cuando ejecutas el código de Fibonacci de manera secuencial, cada cálculo se realiza uno tras otro, sin aprovechar la capacidad de procesamiento paralelo de la máquina. En cambio, cuando utilizas MPI, divides el trabajo entre múltiples procesos, cada uno de los cuales puede realizar sus propios cálculos en paralelo.
