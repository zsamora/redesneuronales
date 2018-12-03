Para ejecutar el programa, basta con ingresar el siguiente comando en consola:

>$ python3 geneticalgorithm.py

Esto ejecutará el algoritmo genético para resolver el problema de N-queen con
los valores globales definidos al inicio del código. Las variables (para que
puedan modificarse fácilmente son:

  - N = Tamaño del tablero (N x N), correspondiente a la codificación
        recomendada del tablero de una lista de largo N, tal que cada elemento
        de la lista corresponde a la posición de la reina en esa columna
  - P = Tamaño de la población (inicial como de las posteriores)
  - T = Tamaño del torneo (se utiliza un valor recomendado de 3/4 * P)
  - M_rate = Mutation rate, para realizar la mutación en los hijos
  - MAX_gen = Máximo número de generaciones para evitar que el algoritmo
              corra infinitamente

Para los ejemplos resueltos en clase, los valores corresponden a su análogo,
como por ejemplo N es el largo de la secuencia de bits o el largo de la
palabra, mientras que todo el resto se mantiene igual. Si se necesita
información más detallada del proceso dentro de geneticAlgorithmBits,
geneticAlgorithmQueen y geneticAlgorithmWord se encuentran cuatro print que
muestran el proceso de selección de padres genéticos y la posterior nueva
población creada. La solución de los tres problemas NO necesariamente se va a
encontrar, y esto dependerá de la naturaleza aleatoria de la creación de la
población, la mutación y la elección de la secuencia/palabra secreta. Si desea
observar el caso en que funcione directamente, se recomienda utilizar valores
pequeños para N (mayor o igual a 4 para el problema N-Queen).
