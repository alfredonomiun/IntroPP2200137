Prácticas CUDA
CUDA representa las siglas de Arquitectura Unificada de Dispositivos de Cómputo (Compute Unified Device Architecture), refiriéndose a una plataforma de computación en paralelo. Incluye un compilador y un conjunto de herramientas de desarrollo diseñadas por Nvidia que habilitan a los programadores para utilizar una variante del lenguaje de programación C (CUDA C) con el fin de codificar algoritmos para las GPU de Nvidia.


Ejecución de CUDA en el supercomputador GUANE
Cuando nos encontramos en GUANE, solicitamos recursos con el comando:
"srun -n 8 --pty /bin/bash"
En este caso, estamos utilizando una máquina con 8 nodos que poseen la máxima cantidad de GPUs disponibles. Si necesitamos un número específico de GPUs, utilizamos el siguiente comando:
"srun -n 8 --gres=gpu:2 --pty /bin/bash"
Aquí estamos solicitando el uso de 2 GPUs.
Si deseamos acceder a otra máquina como Yaje, empleamos el siguiente comando:
"srun -p Viz -n 2 --pty /bin/bash"
Una vez en una de las particiones o máquinas, cargamos los módulos necesarios para ejecutar nuestros códigos en CUDA:
"module load devtools/cuda/8.0"
Luego, solo queda compilar y ejecutar nuestros códigos:

"nvcc XXX.cu -o Exec"
"./Exec"


Detalles sobre el primer código (pt1.cu)
Fundamentalmente, este programa utiliza CUDA para aprovechar la capacidad de procesamiento en paralelo de una GPU y realizar la suma de dos arreglos de manera eficiente.
Se definen constantes como NB (número de bloques), NT (número de hilos por bloque) y N (tamaño total del arreglo), que desempeñan un papel crucial al realizar pruebas bajo diversas condiciones de recursos asignados.

Información acerca del segundo código (multidevice.cu)
Este código, un programa en C que utiliza CUDA y multihilo, calcula el producto punto de dos vectores en paralelo utilizando varios dispositivos GPU.
Realiza esta tarea utilizando dos dispositivos GPU y emplea la biblioteca book.h para gestionar errores de CUDA y proporcionar funciones para trabajar con hilos.

Optimizaciones en pt1.cu
En este código, se identifica un bucle for que no aporta funcionalidad y ralentiza el tiempo de ejecución. Al comentar o eliminar esta línea de código, donde se asigna este bloque de iteración, se aprecia una mejora en el tiempo de ejecución sin afectar el resultado de la suma.

También se presenta una solución que hace uso de la memoria unificada y la versión de la librería CUDA que lo permite, obteniendo resultados similares a la primera mejora.

Mejoras en multidevice.cu
La mejora en el código de multidevice.cu se centra en la posibilidad de agregar dinámicamente la cantidad de GPUs que ejecutarán el cálculo y dividir los datos de manera uniforme entre estas GPUs.
Esto se logra mediante una reestructuración del código sin afectar la lógica del proceso.
