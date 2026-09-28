# Estructura rítmica y temporal en las codas del cachalote: informe de las fases 1 y 2

**Proyecto:** *Discovering Hidden Structures in Sperm Whale Vocalizations Using Unsupervised Machine Learning*
**Repositorio:** [github.com/amaisuarez/whale-vocalization-patterns](https://github.com/amaisuarez/whale-vocalization-patterns)
**Fecha:** 26 de septiembre de 2026
**Estado:** fases 1 (replicación) y 2 (tempo en el repertorio) completadas

---

## Resumen

Este informe recoge las dos primeras fases de una investigación personal sobre la comunicación del cachalote (*Physeter macrocephalus*) mediante aprendizaje automático no supervisado. Se analizaron 7.268 codas del clan vocal EC1 de Dominica, procedentes del conjunto de datos anotado del Dominica Sperm Whale Project (Sharma et al., 2024).

En la **fase 1** se validó el método comprobando si un algoritmo de agrupamiento, sin acceso a las etiquetas de los expertos, recuperaba los tipos de coda conocidos. Los grupos obtenidos resultaron casi puros (homogeneidad 0,96) y más finos que las categorías de los expertos (completitud 0,68). La principal discrepancia fue la división del tipo más frecuente, `1+1+3`, en tres clases de duración (≈ 0,81, 1,05 y 1,28 s), coherente con la noción de *tempo* descrita por Sharma et al. (2024).

En la **fase 2** se investigó si estas clases de tempo son una propiedad general del repertorio. Se introdujeron dos salvaguardas metodológicas: una medida de separación entre modos (D de Ashman) y un *bootstrap* por sesiones de grabación que corrige la falta de independencia entre codas de una misma conversación. Con estos controles, `1+1+3` es el único tipo con clases de tempo robustas. Cuatro divisiones aparentes en otros tipos resultaron ser artefactos del modelo. Se rechazó además la independencia entre ritmo y tempo, y no se encontró apoyo para la hipótesis de que el tempo varíe de forma conjunta entre tipos de coda dentro de una misma sesión.

El siguiente paso propuesto es modelar cómo cambia la forma rítmica de una coda cuando cambia su velocidad, lo que conecta con el fenómeno del *rubato*.

---

## 1. Introducción y motivación

El proyecto nació a raíz del documental *Fathom: Decoding the Deep* y del interés por la intersección entre ciencia marina, inteligencia artificial y análisis de datos. La pregunta de investigación general es:

> **¿Pueden las técnicas de aprendizaje automático no supervisado revelar estructuras y patrones significativos en las vocalizaciones del cachalote?**

El proyecto no pretende demostrar que los cachalotes tengan un lenguaje. Su objetivo es más acotado y medible: identificar si existen agrupaciones, similitudes o estructuras recurrentes que puedan detectarse con métodos computacionales, y documentar el proceso de forma transparente, incluidos los experimentos que no funcionan.

## 2. Contexto biológico

Los cachalotes hembra viven en **unidades sociales**: grupos familiares estables formados por hembras emparentadas y sus crías. Las unidades que comparten un mismo repertorio vocal forman un **clan vocal**, una especie de dialecto que se transmite culturalmente (Rendell y Whitehead, 2003). Clanes distintos pueden compartir territorio, pero las unidades tienden a asociarse con las de su propio clan. En Dominica se conocen dos clanes principales, **EC1** y **EC2** (Gero et al., 2016).

Los cachalotes se comunican mediante **codas**: secuencias breves de entre 3 y 10 clics con un patrón temporal característico. La información de una coda reside principalmente en los **intervalos entre clics** (ICI, *inter-click intervals*), no en el timbre de cada clic. Los expertos clasifican las codas en **tipos** según su patrón. Por ejemplo, `1+1+3` designa un clic, una pausa, otro clic, otra pausa y tres clics rápidos, y `5R` designa cinco clics regulares.

Sharma et al. (2024) describieron las codas como un sistema combinatorio con cuatro rasgos: **ritmo** (la forma del patrón), **tempo** (la duración total), **rubato** (ajustes graduales de velocidad entre codas consecutivas) y **ornamentación** (un clic adicional dependiente del contexto).

## 3. Cambio de enfoque: del espectrograma al tiempo

La metodología inicial del proyecto preveía extraer descriptores acústicos (frecuencia, energía, rasgos espectrales) a partir de grabaciones de audio, que es el enfoque habitual en bioacústica. Al revisar la literatura quedó claro que, en el caso de las codas, **la estructura principal está en la temporización**. Por ello se decidió empezar con datos de intervalos ya anotados. Esto permite trabajar sin procesar audio y con etiquetas de referencia para validar el método. El análisis espectral queda como una línea futura.

## 4. Datos

Se utilizó el conjunto de codas anotadas del Dominica Sperm Whale Project, publicado junto con Sharma et al. (2024) bajo licencia CC BY 4.0. El conjunto contiene **8.719 codas** grabadas entre **2005 y 2016**. De cada coda se conoce el número de clics, la duración, los ICIs, el tipo asignado por los expertos, el clan, la unidad social y, cuando se pudo identificar, la ballena individual. Los datos se descargan con un script que fija una versión concreta del repositorio de origen, para garantizar la reproducibilidad.

### 4.1. Controles de calidad

El archivo es internamente coherente: los ICIs suman la duración declarada y el número de intervalos coincide con el número de clics menos uno (salvo 3 filas de 8.719). Se detectaron varias características que condicionan el análisis:

| Característica | Valor | Implicación |
|---|---|---|
| Codas del clan EC1 / EC2 | 7.770 / 949 | Dos dialectos mezclados en el mismo archivo |
| Codas etiquetadas como `*-NOISE` | 600 | No clasificables; se excluyen |
| Codas sin ballena identificada | 5.752 (66 %) | Los análisis por individuo solo pueden usar un tercio de los datos |
| Formato de fechas | Mezcla de `/` y `-`, siempre día primero | Corregido en la carga |
| Peso del tipo `1+1+3` | 49 % del subconjunto de análisis | Fuerte desequilibrio entre tipos |

![Número de codas por número de clics y por tipo](figures/01_counts.png)

### 4.2. Subconjunto de análisis

Se trabajó con el clan **EC1** (el analizado por Sharma et al.), excluyendo las codas `*-NOISE` y las de menos de 3 clics. El resultado son **7.268 codas de 24 tipos**, producidas por **10 unidades sociales** en **103 sesiones** de grabación (91 días distintos). Aparecen 24 tipos, y no los 21 que se suelen citar, porque EC1 contiene unas pocas codas de tipos característicos de EC2.

## 5. Metodología general

### 5.1. Representación de las codas

Cada coda se describe con dos elementos:

- **Ritmo:** los ICIs divididos por la duración total. Captura la forma del patrón con independencia de la velocidad.
- **Tempo:** la duración total, expresada en escala logarítmica porque las variaciones de velocidad son proporcionales.

### 5.2. Agrupamiento

Las codas se agrupan **por separado dentro de cada número de clics**. El número de clics es una propiedad observable directamente, así que no tiene sentido pedir al algoritmo que la redescubra. Además, así se evita comparar vectores de longitud distinta.

Se emplean **modelos de mezcla gaussiana** (GMM), que describen los datos como la suma de varias distribuciones con forma de campana. El número de grupos se elige con el **criterio de información bayesiano** (BIC; Schwarz, 1978), que equilibra el ajuste a los datos con la complejidad del modelo. **Las etiquetas de los expertos nunca se usan para elegir el número de grupos**; solo se usan después, para evaluar.

### 5.3. Evaluación

Se usan cuatro métricas de comparación con las etiquetas:

| Métrica | Vale 1 cuando… | No penaliza… |
|---|---|---|
| Homogeneidad | cada grupo contiene un solo tipo | dividir un tipo en varios grupos |
| Completitud | cada tipo cae en un solo grupo | fusionar varios tipos en un grupo |
| NMI | ambas (media armónica; Rosenberg y Hirschberg, 2007) | — |
| ARI | acuerdo por pares, corregido por azar (Hubert y Arabie, 1985) | — |

Separar homogeneidad y completitud resultó decisivo, porque distingue un método que *se equivoca* de uno que ve distinciones *más finas* que los expertos.

## 6. Fase 1: replicación

**Pregunta:** ¿recupera un método no supervisado los tipos de coda definidos por los expertos?

Replicar primero cumple una función de validación: si el método no recupera la estructura conocida, nada de lo que "descubra" después será fiable.

### 6.1. Exploración

El análisis exploratorio mostró que tipos como `5R1`, `5R2` y `5R3` tienen ritmos muy parecidos y se distinguen sobre todo por su duración. También mostró que `1+1+3` tiene una distribución de duraciones inusualmente amplia y con varios picos.

![Ritmo y tempo de las codas de 5 clics](figures/01_rhythm_tempo_5click.png)

![Distribución de duraciones de las codas 1+1+3](figures/01_113_duration.png)

### 6.2. Resultados del agrupamiento

| Representación | ARI | NMI | Homogeneidad | Completitud | Grupos |
|---|---|---|---|---|---|
| Solo ritmo | 0,43 | 0,71 | 0,84 | 0,61 | 28 |
| **Ritmo + tempo** | **0,49** | **0,79** | **0,96** | **0,68** | 34 |
| ICIs sin normalizar | 0,48 | 0,79 | 0,95 | 0,68 | 36 |

Añadir el tempo mejora claramente los resultados. Con ritmo y tempo, los grupos son **casi puros** (homogeneidad 0,96), pero hay 34 grupos para 24 tipos: el algoritmo divide más que los expertos.

### 6.3. Dónde discrepan el algoritmo y los expertos

En las codas de 5 clics se observaron tres tipos de discrepancia:

1. **División por tempo:** `1+1+3` se reparte en subgrupos puros con duraciones medias claramente distintas.
2. **División sin diferencia de tempo:** `5R1` se divide en tres grupos con la misma duración media (≈ 0,33 s), que difieren en detalles finos del ritmo.
3. **Fusión:** un grupo reúne `5R2`, casi todas las `5R3` (solo 19 en EC1) y algunas `5R1`.

![División de 1+1+3 por el agrupamiento no supervisado](figures/02_113_split.png)

### 6.4. Comprobaciones

Para descartar que la división de `1+1+3` fuera un artefacto de las condiciones de grabación, se midió su asociación con la unidad social y con el año. Ambas fueron débiles (NMI ≈ 0,05), y todas las unidades producen codas en todas las clases de tempo, aunque con preferencias distintas.

La estabilidad frente a la semilla aleatoria fue **moderada** (ARI 0,68–0,77 entre ejecuciones). Dentro de `1+1+3`, las fronteras entre subgrupos cambiaban notablemente de una ejecución a otra (ARI 0,53–0,61). El modelo completo detectaba *que* había que dividir `1+1+3`, pero no *dónde* de forma fiable.

### 6.5. Test enfocado

Se planteó la pregunta de forma directa: un modelo de mezcla sobre la duración de las codas `1+1+3` únicamente. El resultado fue **k = 3 en las 5 semillas** y en 30 de 30 remuestreos *bootstrap*, con centros en **0,81, 1,05 y 1,28 s**. La ventaja en BIC frente a las alternativas fue clara: +109 puntos frente a 2 clases y +33 frente a 4.

![Tres clases de tempo en 1+1+3](figures/02_113_tempo_classes.png)

**Conclusión de la fase 1:** el método recupera la estructura conocida y, sin supervisión, reproduce un resultado publicado y no evidente: la existencia de clases discretas de tempo. La fase también dejó una lección metodológica: un modelo amplio sirve para *detectar* estructura, pero *confirmarla* requiere una prueba específica.

## 7. Fase 2: el tempo en el conjunto del repertorio

**Pregunta:** ¿las clases de tempo son exclusivas de `1+1+3` o son una dimensión compartida por el repertorio, con clases que se alinean entre tipos?

### 7.1. Ritmo y tempo no son independientes

El plan inicial era agrupar los tipos con el mismo ritmo y distinta velocidad (por ejemplo, `5R2` y `5R3`) en familias rítmicas, y buscar clases de tempo dentro de cada familia. Para comprobar qué tipos comparten ritmo se entrenó, para cada par de tipos con el mismo número de clics, un clasificador que solo veía el ritmo, sin la duración. Si dos tipos compartieran ritmo, el clasificador acertaría al azar (50 %).

No fue así. Incluso los pares más parecidos se distinguen solo por el ritmo muy por encima del azar: `5R2` frente a `5R3` al 70 %, `4R1` frente a `4R2` al 78 % y `5R1` frente a `5R3` al 79 %. **La forma de una coda cambia ligeramente con su velocidad.** Como cualquier criterio para fusionar tipos habría sido arbitrario, se reformuló la pregunta: analizar cada tipo por separado y comparar después los resultados.

### 7.2. Dos correcciones al método

**Separación real entre modos.** El BIC tiende a ajustar varias gaussianas a una distribución asimétrica aunque tenga un solo pico. Para detectarlo se usó la **D de Ashman** (Ashman et al., 1994), que mide la distancia entre dos picos en relación con su anchura: D > 2 indica picos separados y D < 1 indica un único pico partido artificialmente.

**Independencia de las observaciones.** Las codas de una misma unidad grabadas el mismo día forman parte de la misma conversación y no son independientes. Tratarlas como si lo fueran es un caso de **pseudorreplicación** (Hurlbert, 1984) y sobreestima la certeza. Por ello, la robustez se evaluó con un ***bootstrap* por sesiones**: se remuestrean sesiones completas (unidad × día), no codas sueltas.

### 7.3. Resultados por tipo de coda

Se analizaron los 18 tipos con al menos 30 codas. Se clasificó como **robusto** un tipo con D ≥ 2, multimodal en al menos el 90 % de los remuestreos por sesión y con todas sus clases de peso apreciable (≥ 10 %). Se clasificó como **candidato** un tipo con D ≥ 1,5 y multimodal en al menos el 75 % de los remuestreos.

| Tipo | Codas | Sesiones | Centros (s) | D mín. | Multimodal (bootstrap sesiones) | Veredicto |
|---|---|---|---|---|---|---|
| `1+1+3` | 3.574 | 73 | 0,81 / 1,05 / 1,28 | 2,42 | 100 % | **Robusto** |
| `5R1` | 1.510 | 75 | 0,33 / 0,33 / 0,35 | 0,01 | 100 % | Artefacto |
| `4D` | 219 | 24 | 0,52 / 0,53 | 0,53 | 69 % | Artefacto |
| `8i` | 178 | 49 | 0,34 / 0,37 | 0,25 | 100 % | Artefacto |
| `3D` | 61 | 9 | 0,31 / 0,33 | 0,55 | 46 % | Artefacto |
| `6i` | 182 | 52 | 0,23 / 0,40 / 0,51 | 1,74 | 96 % | Candidato |
| `7i` | 145 | 50 | 0,21 / 0,32 / 0,50 | 2,39 | 87 % | Candidato |
| `9i` | 135 | 44 | 0,19 / 0,37 / 0,67 | 4,35 | 100 % | Candidato (una clase de solo 5 codas) |
| `7D1` | 171 | 17 | 1,08 / 1,30 | 2,36 | 78 % | Candidato (82 % de una unidad) |
| `4R1`, `1+32` | — | — | — | — | 40–50 % | Sin apoyo |
| `4R2` | 303 | 14 | 0,97 | — | 59 % | Tempo único, ambiguo (97 % de una unidad) |
| `5R2`, `8R`, `7D2`, `10i`, `2+3`, `1+31` | — | — | un centro | — | — | Tempo único |

Con el nuevo método, `1+1+3` sigue siendo robusto: es multimodal en el 100 % de los remuestreos por sesión y tiene 3 clases en el 90 %, frente al 100 % que daba el *bootstrap* por codas de la fase 1. **Sin la D de Ashman se habrían comunicado cuatro falsos descubrimientos.**

### 7.4. La escala de tempos del repertorio

Las duraciones del repertorio abarcan un orden de magnitud (de ≈ 0,15 a ≈ 1,6 s), y muchos tipos quedan entre las clases de `1+1+3`. No se observa una rejilla pequeña de tempos compartida por todo el repertorio. Con un solo tipo robustamente multimodal, tampoco hay datos suficientes para contrastarlo formalmente. Queda anotada, sin afirmarse, una observación: los tipos más lentos (`8R`, el modo lento de `7D1` y la clase lenta de `1+1+3`) se concentran en torno a 1,2–1,3 s.

![Duración por tipo de coda](figures/03_tempo_ladder.png)

### 7.5. Una coincidencia sugerente: `7D1` y `1+1+3`

`7D1` (7 clics, ritmo decreciente) presenta dos modos cuyos intervalos de confianza se solapan con dos de las clases de `1+1+3`:

| | Clase media | Clase lenta |
|---|---|---|
| `1+1+3` (IC 95 %) | 1,033 – 1,070 s | 1,267 – 1,288 s |
| `7D1` (IC 95 %) | 1,030 – 1,122 s | 1,264 – 1,344 s |

Sería una coincidencia notable entre ritmos y números de clics distintos, pero la evidencia es **débil**: el 82 % de las codas `7D1` proceden de la unidad F, y 127 de esas 141 se grabaron en solo 3 días. La hipótesis de clases de tempo compartidas entre tipos queda abierta.

### 7.6. Hipótesis descartada: el "tempo de la conversación"

Al revisar los datos por día, se observó que en ciertas sesiones tanto `7D1` como `1+1+3` sonaban mayoritariamente lentas, y en otras, rápidas. De ahí surgió la hipótesis de que el tempo sube y baja de forma conjunta en todos los tipos de coda dentro de una sesión.

Se contrastó en las 40 sesiones con datos suficientes, correlacionando el tempo medio de `1+1+3` con el del resto de tipos (tras estandarizar la duración dentro de cada tipo). El resultado fue **r = 0,15, p = 0,37**, y un test de permutación dio el mismo resultado. En el análisis tipo a tipo, dos comparaciones alcanzaron p < 0,05, pero con signos opuestos, y ninguna superó la corrección de Bonferroni para 8 comparaciones (umbral 0,0063). **La hipótesis no se sostiene**: el patrón inicial era anecdótico.

![Covariación del tempo por sesión](figures/03_session_covariation.png)

## 8. Síntesis de la evidencia

| Afirmación | Estado |
|---|---|
| Un método no supervisado recupera los tipos de coda de EC1 | **Confirmado** (homogeneidad 0,96) |
| El tempo es necesario para separar tipos de ritmo similar | **Confirmado** |
| `1+1+3` tiene tres clases de tempo (≈ 0,81 / 1,05 / 1,28 s) | **Robusto** |
| Esas clases no se explican por la unidad social o el año | **Confirmado** (NMI ≈ 0,05) |
| Ritmo y tempo son rasgos independientes | **Rechazado** |
| Clases de tempo en `5R1`, `4D`, `8i`, `3D` | **Artefactos** |
| Los tipos `i` (`6i`, `7i`, `9i`) tienen varias clases de tempo | **Candidato** |
| `7D1` comparte clases de tempo con `1+1+3` | **Débil, con factor de confusión** |
| Todo el repertorio comparte una rejilla de tempos | **No contrastable** con estos datos |
| El tempo covaría entre tipos dentro de una sesión | **Sin apoyo** |

## 9. Discusión

### 9.1. Relación con la literatura

El resultado central, las clases discretas de tempo dentro de `1+1+3`, es coherente con la descripción de Sharma et al. (2024) del tempo como rasgo categórico. Su valor en este proyecto no está en la novedad, sino en la **validación**: un método sencillo y sin supervisión llega por sí solo a la misma conclusión que un estudio especializado, lo que da confianza para los pasos siguientes.

El rechazo de la independencia entre ritmo y tempo es relevante para cualquier trabajo que trate ambos rasgos por separado. Si la forma de una coda depende de su velocidad, la forma exacta de esa dependencia es una pregunta en sí misma.

### 9.2. Lecciones metodológicas

1. **Replicar antes de descubrir.** Validar con estructura conocida da una referencia objetiva para juzgar el método.
2. **Una sola métrica puede ocultar lo importante.** El ARI por sí solo habría presentado como un resultado mediocre lo que en realidad era una distinción más fina que la de los expertos.
3. **El BIC tiende a sobredividir.** Hace falta verificar que los modos están realmente separados.
4. **La unidad de independencia es la sesión, no la coda.** Ignorarlo produce una confianza injustificada.
5. **Detectar y confirmar son pasos distintos,** y conviene usar herramientas distintas para cada uno.
6. **Los resultados negativos son resultados.** Dos hipótesis plausibles se contrastaron y se descartaron, y quedan documentadas.

## 10. Limitaciones

- **Un único clan y una única población.** Los resultados se refieren a EC1 en Dominica; su generalidad no está comprobada.
- **Pocas sesiones en varios tipos.** Tipos como `7D1`, `4R2` o `1+32` dependen casi por completo de una sola unidad o de pocos días.
- **Identificación individual incompleta.** Dos tercios de las codas no tienen ballena identificada, así que no se puede descartar que ciertas clases de tempo reflejen estilos individuales.
- **Etiquetas como referencia.** Las etiquetas de los expertos se usan como referencia, pero también son una construcción humana. Las discrepancias pueden deberse tanto al método como a las etiquetas.
- **Solo temporización.** No se ha analizado todavía el contenido espectral de los clics.
- **Supuestos del modelo.** Los GMM suponen grupos gaussianos. Otras familias de modelos podrían dar particiones distintas en los casos dudosos.

## 11. Conclusiones

1. El conjunto de datos del DSWP es fiable y adecuado para análisis no supervisados, siempre que se tengan en cuenta la mezcla de clanes, el desequilibrio entre tipos y la estructura por sesiones.
2. Un agrupamiento sencillo basado en ritmo y tempo recupera la organización de las codas de EC1 descrita por los expertos, y la afina.
3. La coda `1+1+3`, identitaria del clan EC1, se produce en **tres velocidades discretas**. Es el resultado más sólido del proyecto hasta ahora y resiste todas las comprobaciones aplicadas.
4. Ese patrón **no se generaliza de forma demostrable** al resto del repertorio con los datos disponibles. Hay candidatos interesantes, pero ninguno supera todos los controles.
5. Ritmo y tempo **interactúan**: la forma de una coda cambia con su velocidad.
6. Se ha construido una base de código reproducible y documentada sobre la que apoyar las fases siguientes.

## 12. Próximos pasos

**Fase 3 (propuesta): interacción entre ritmo y tempo.** Modelar cómo cambia la forma de una coda cuando cambia su velocidad. La pregunta es si, cuando una coda se ralentiza, todos los intervalos se estiran por igual o algunos cambian más que otros. `1+1+3` es el candidato natural, por su tamaño y sus tres clases de tempo bien definidas. Esta línea conecta con el *rubato* descrito por Sharma et al. (2024).

**Líneas complementarias:**

- **Preferencias de tempo por unidad social.** Contrastar formalmente si las unidades difieren en el uso de las tres clases de `1+1+3`, tomando la sesión como unidad de análisis (por ejemplo, con modelos de efectos mixtos o permutaciones por sesión).
- **Validación fuera de muestra con EC2.** Repetir todo el análisis con las 949 codas del otro clan.
- **La división de `5R1`.** Estudiar si los tres subgrupos de igual duración son estables y si se asocian a unidades, individuos o contexto.
- **Secuencias de codas.** El conjunto `sperm-whale-dialogues.csv`, ya descargado, contiene codas con marca temporal y hablante, lo que permitiría estudiar intercambios entre ballenas.
- **Análisis espectral.** Incorporar grabaciones de audio para estudiar la forma de onda de los clics, en línea con el objetivo original del proyecto.

## 13. Reproducibilidad

Todo el análisis es reproducible desde el repositorio:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
jupyter lab
```

| Recurso | Contenido |
|---|---|
| [`notebooks/01_exploration.ipynb`](../notebooks/01_exploration.ipynb) | Exploración y controles de calidad |
| [`notebooks/02_replication_clustering.ipynb`](../notebooks/02_replication_clustering.ipynb) | Fase 1: replicación |
| [`notebooks/03_tempo_across_repertoire.ipynb`](../notebooks/03_tempo_across_repertoire.ipynb) | Fase 2: tempo en el repertorio |
| [`src/whalecodas/`](../src/whalecodas/) | Código reutilizable: carga, rasgos, agrupamiento y tempo |
| [`journal/`](../journal/) | Diario de investigación, con decisiones y errores |

## 14. Referencias

- Ashman, K. M., Bird, C. M. y Zepf, S. E. (1994). Detecting bimodality in astronomical datasets. *The Astronomical Journal*, 108, 2348–2361.
- Gero, S., Whitehead, H. y Rendell, L. (2016). Individual, unit and vocal clan level identity cues in sperm whale codas. *Royal Society Open Science*, 3, 150372.
- Hubert, L. y Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2, 193–218.
- Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54, 187–211.
- Rendell, L. E. y Whitehead, H. (2003). Vocal clans in sperm whales (*Physeter macrocephalus*). *Proceedings of the Royal Society B*, 270, 225–231.
- Rosenberg, A. y Hirschberg, J. (2007). V-Measure: a conditional entropy-based external cluster evaluation measure. *Proceedings of EMNLP-CoNLL*, 410–420.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6, 461–464.
- Sharma, P., Gero, S., Payne, R., Gruber, D. F., Rus, D., Torralba, A. y Andreas, J. (2024). Contextual and combinatorial structure in sperm whale vocalisations. *Nature Communications*, 15, 3617. https://doi.org/10.1038/s41467-024-47221-8

**Datos:** Dominica Sperm Whale Project, publicados con Sharma et al. (2024) bajo licencia CC BY 4.0. Repositorio: https://github.com/pratyushasharma/sw-combinatoriality · Archivo: https://doi.org/10.5281/zenodo.10817697

---

## Anexo: glosario

| Término | Definición |
|---|---|
| Clan vocal | Conjunto de unidades sociales que comparten dialecto (repertorio de codas) |
| Unidad social | Grupo familiar estable de hembras emparentadas y sus crías |
| Coda | Secuencia corta de clics con un patrón temporal característico |
| ICI | Intervalo de tiempo entre dos clics consecutivos |
| Ritmo | ICIs normalizados por la duración total; la forma de la coda |
| Tempo | Duración total de la coda; su velocidad |
| Sesión | Una unidad social grabada en un día concreto |
| GMM | Modelo de mezcla gaussiana; describe los datos como suma de campanas |
| BIC | Criterio para elegir el número de grupos, que penaliza la complejidad |
| Homogeneidad / completitud | Pureza de los grupos / unidad de cada tipo en un solo grupo |
| *Bootstrap* | Remuestreo repetido de los datos para medir la estabilidad de un resultado |
| Pseudorreplicación | Tratar como independientes observaciones que no lo son |
| D de Ashman | Medida de separación entre dos picos; D > 2 indica picos reales |
| Corrección de Bonferroni | Ajuste del umbral de significación cuando se hacen varias pruebas |
