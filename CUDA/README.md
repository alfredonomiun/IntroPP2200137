LO PRIMERO QUE VAMOS A HACER ES EXPLICAR LOS CODIGOS ORIGINALES Y DESPUES EXPLICAR LAS MEJORAS QUE SE LE HICIERON

## vectorAdd.cu

Este código es un ejemplo básico de programación en CUDA para la adición de vectores en paralelo, empezando tenemos :

1-Definición del Kernel CUDA:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/e86b24bc-3c5b-4339-98d1-ce22736b7a13)

Este kernel se ejecuta en paralelo en la GPU. Cada hilo (thread) se encargará de sumar un elemento correspondiente de los vectores A y B y almacenar el resultado en el vector C.

2-Funcion principal ( Host )

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/4ac29213-a363-46ac-ab8c-e857946d097c)

La función principal realiza las siguientes tareas:
*Aloja memoria en el host para los vectores h_A, h_B y h_C.
*Inicializa los vectores h_A y h_B con valores aleatorios.
*Aloja memoria en el dispositivo (GPU) para los vectores d_A, d_B y d_C.
*Copia los datos desde el host al dispositivo.
*Lanza el kernel CUDA vectorAdd.
*Copia los resultados desde el dispositivo de nuevo al host.
*Verifica la precisión de los resultados.
*Libera la memoria.

3- Aloja y copia memoria en el dispositivo:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/e3883993-bebd-461d-8804-9b3aa27b44df)

Se aloja memoria en el dispositivo para los vectores de entrada (d_A, d_B) y el vector de salida (d_C).

4- Copia de datos entre el host y el dispositivo:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/ebbb7a55-658a-4e4e-9bcd-0134a9cfd329)

Los datos de los vectores de entrada se copian desde el host al dispositivo antes de ejecutar el kernel.

5- Lanzamiento del kernel:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/0cf36511-4c6f-4654-9f32-a3dde30a86ab)

Se lanza el kernel vectorAdd con una configuración de hilos y bloques específica.

6- Verificación de los resultados:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/d35def3e-1ff8-4913-aa5f-49a2fd5cd37c)

Se verifica que la suma de cada par de elementos de h_A y h_B sea igual al correspondiente elemento de h_C.

En resumen, este código realiza una adición de vectores en paralelo utilizando CUDA, con manejo de errores y verificación de resultados.

## Vector.cu

Esta aplicación en CUDA realiza la adición de dos matrices bidimensionales (a y b) y almacena el resultado en otra matriz bidimensional c.

Entonces, una explicacion detallada seria: 

1- Definición del Kernel CUDA:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/b0aca5e9-036c-49b2-8833-25711a0c6851)

Este kernel se ejecuta en la GPU y realiza la adición de dos elementos correspondientes de las matrices a y b, almacenando el resultado en la matriz c. El índice i se calcula para acceder a los elementos individuales de las matrices.

2- funcion principal(host) 
La función principal realiza las siguientes tareas:
*Alojaa memoria en el dispositivo para las matrices a, b y c.
*Inicializa las matrices a y b en el host.
*Copia los datos desde el host al dispositivo.
*Lanza el kernel CUDA add.
*Copia los resultados desde el dispositivo de nuevo al host.
*Imprime los resultados.

3- Aloja y copia memoria en el dispositivo:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/3c859d8b-d1a8-48f1-9300-7bc6300c657e)

*Se aloja memoria en el dispositivo para las matrices dev_a, dev_b y dev_c.

4- Ciclos para llenar las matrices en el host

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/7bb28fff-2486-4af7-abc7-d046cf2e8cc4)

Las matrices a y b se llenan con valores.

5-Copia de datos entre el host y el dispositivo:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/95b3c6d5-f50b-4a58-a671-e9f2c9a40a1a)

Los datos de las matrices a y b se copian desde el host al dispositivo antes de ejecutar el kernel.

6- Lanzamiento del kernel:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/a519e1db-6151-4c04-a47c-b078116c67aa)

Se lanza el kernel add con una configuración de bloques de tamaño (COLUMNS, ROWS).

7-Copia de datos desde el dispositivo al host:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/45ee975b-555e-4830-af0f-897aeb0c3d0f)

Los resultados se copian desde el dispositivo al host después de ejecutar el kernel.

8-Imprimir resultados

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/b9629aa2-664e-4d83-a474-13758ead30fe)

Se imprimen los resultados de la matriz c.

En resumen, este código realiza la adición de dos matrices bidimensionales en paralelo utilizando CUDA, con manejo de memoria en el dispositivo y verificación visual de los resultados.

## Book.h
Este código proporciona algunas funciones y definiciones útiles para trabajar con CUDA y también contiene funciones para manipular hilos en sistemas Windows y POSIX. Aquí hay una explicacion de las partes clave del código:


1- Manejo de errores:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/40bfe9b0-2fbf-44ca-b95c-db5bb975443f)

Se define una función HandleError que imprime mensajes de error de CUDA junto con el nombre del archivo y el número de línea. Esto ayuda en la detección y manejo de errores durante la ejecución.

2-Macros para Manejo de Errores:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/1efb1f1d-e48a-46d6-8963-eb111670a368)

Se definen macros HANDLE_ERROR y HANDLE_NULL para facilitar la comprobación de errores y la gestión de la memoria.

3-Funciones de Manipulación de Hilos:

Se proporcionan funciones para crear, esperar y destruir hilos. Esto es útil para trabajar con programación paralela y multi-hilo.

4-Funciones para Generar Datos Aleatorios:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/81395f6d-bf07-4764-b670-5004584ac32f)

Se proporcionan funciones para generar bloques grandes de datos aleatorios.

5-Kernels CUDA para Convertir Números de Punto Flotante a Colores:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/f2ab9370-10de-4699-b54d-1d29521ddaac)

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/dc5432dc-19f1-4795-b37c-5ae4d62c8f8a)

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/951aebdd-46c2-478a-afe3-1267f20bd526)

Se definen kernels CUDA para convertir números de punto flotante a colores. Estos kernels son utilizados para realizar operaciones en la GPU.

6-Definiciones de Tipos de Datos:

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/ba0e0a87-7a14-4fdf-be75-011d59cc7bd2)

Se definen tipos de datos para hilos dependiendo del sistema operativo (Windows o POSIX).

7- Funciones para Manipulación de Hilos (Windows y POSIX):

![image](https://github.com/alfredonomiun/IntroPP2200137/assets/94908591/f9f605de-0d55-46cf-bd41-fefd454375dc)

Se proporcionan funciones para la creación, espera y destrucción de hilos, dependiendo del sistema operativo.

En resumen, este código proporciona utilidades y funciones comunes que pueden ser útiles al trabajar con CUDA y programación multi-hilo. También se incluyen funciones para generar datos aleatorios y realizar operaciones en la GPU mediante kernels CUDA.

## AHORA EXPLICARE LAS MEJORAS


## MEJORA PRIMER CODIGO vectoradd_M
1-Se crearon funciones separadas para inicializar, copiar y liberar memoria tanto en el host como en el dispositivo, mejorando la modularidad del código.
2-Se agregaron funciones para manejar errores de CUDA (HANDLE_ERROR) y para imprimir mensajes de error junto con la ubicación en el código.
3-Se utilizó static_cast en lugar de rand()/(float)RAND_MAX para la generación de números aleatorios, que es una práctica más moderna en C++.
4-Se agregó un mensaje de impresión para indicar que la aplicación ha finalizado exitosamente.

## MEJORA SEGUNDO CODIGO vector_M
1-Se eliminaron las asignaciones dinámicas de memoria para dev_a, dev_b, y dev_c ya que no son necesarias. Se cambió la declaración de estas variables a arreglos estáticos.
2-Se creó una función initializeHostArrays para mejorar la legibilidad y mantener el código organizado.
3-Se eliminaron variables innecesarias y se redujo la complejidad del código.
4-Se añadió un comentario para explicar el propósito de cada sección del código.
