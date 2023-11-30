# Innovaciones en el Proyecto

Este software simula la progresión de un campo de temperatura en un dominio bidimensional mediante la aplicación de la ecuación del calor, aprovechando la tecnología MPI para distribuir y agilizar los cálculos. La ecuación del calor, una ecuación diferencial parcial, modela la variación de la temperatura en un punto en función del tiempo y el espacio.

Se aplica en diversas áreas científicas e ingenieriles para modelar la conducción de calor en sistemas físicos. La capacidad de simular la propagación del calor resulta fundamental en la física, la ingeniería y las ciencias de la tierra.


Descripción y explicación de la funcionalidad del código

Está compuesto por diez archivos, los cuales se describen los más relevantes

1. heat.h
Una cabecera que define la estructura de datos para representar el campo de temperatura (field). Cada instancia de field tiene dimensiones (nx, ny) y una matriz de datos (data). Esta estructura se utiliza para representar y manipular el campo de temperatura en el dominio del problema.

2. heat.c
Contiene funciones para simular la evolución del campo de temperatura mediante la ecuación del calor. La función evolve() da un paso en el tiempo de la simulación. Utiliza condiciones de contorno y actualiza el campo de temperatura según la derivada laplaciana. Constituye el núcleo del algoritmo de simulación.

Por otro lado, initialize_field() establece las condiciones iniciales del campo de temperatura. Esto representa probablemente un momento inicial en la simulación. La función  initialize_boundary() establece condiciones de contorno para los bordes del dominio, crucial en simulaciones numéricas para asegurar soluciones estables y físicamente relevantes.

3. main.c
Punto de entrada al programa. Configura e inicializa MPI, lee los parámetros de entrada y establece las condiciones iniciales del campo de temperatura. Lleva a cabo la simulación de la ecuación del calor durante un número especificado de pasos, intercambiando los campos de temperatura después de cada paso, con ocasional impresión del estado. Finalmente, cierra la ejecución de MPI y sale del programa.

4. utility.c
Ofrece varias funciones útiles para trabajar con los campos de temperatura. malloc_2d() y free_2d()  reservan y liberan memoria para una matriz 2D, respectivamente; copy_field() replica el contenido de un campo de temperatura en otro; swap_fields() intercambia datos entre dos campos de temperatura; allocate_field() reserva memoria para un campo de temperatura e inicializa a cero.


Mejoras Globales
Estas actualizaciones buscan perfeccionar el rendimiento y la funcionalidad general del código.

1. Estructura, Claridad y Documentación del Código
Se han alterado el código para mejorar su estructura general, buscando una mayor claridad. Además, se han añadido comentarios a diversas sentencias describiendo su funcionalidad. Por último, se ha organizado la estructura de las funciones para prevenir errores lógicos.

2. Optimizaciones de Rendimiento
Se han modificado algunas líneas de código relacionadas con la gestión de memoria, en términos de asignación y liberación, así como bucles que han reducido cálculos innecesarios. Se han mejorado las configuraciones de MPI para optimizar las capacidades de procesamiento paralelo.

3. Archivo Sbatch Ejecutable
Se introduce un archivo denominado sbatch_file.sbatch, que contiene los comandos esenciales para ejecutar el programa correctamente. Importa módulos y bibliotecas necesarios, siguiendo el orden adecuado de cada instrucción requerida para obtener el resultado final.






