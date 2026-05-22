# MLOps4RT-Edge Comparisons

Repositorio orquestador para comparar pipelines MLOps ya existentes:

- `ts`: pipeline de series temporales.
- `events`: pipeline de ventanas de eventos.
- `uniEvents`: pipeline de eventos univariante por medida.

Este repo no contiene modelos, datasets pesados, dashboards ni base de datos. Solo lee `experiment.yaml`, lanza variantes en los repositorios existentes y guarda trazabilidad ligera en `runs/`.

## Configurar Un Experimento

Edita `experiment.yaml` para a?adir medidas, direcciones, conjuntos de bandas o configuraciones temporales.

Los repositorios y pipelines activos se declaran asi:

```yaml
repos:
  ts: C:/Users/inesv/Desktop/STRAST/mlops4rtedgeTS
  events: C:/Users/inesv/Desktop/STRAST/mlops4rtedge
  uniEvents: C:/Users/inesv/Desktop/STRAST/mlops4rtedgeUni

pipelines:
  - ts
  - events
  - uniEvents
```

Las medidas se definen en:

```yaml
grid:
  measures:
    - Battery_Active_Power
```

Las direcciones se definen en:

```yaml
grid:
  directions:
    - high
    - low
```

## Configuraciones Temporales

Puedes probar varias combinaciones de ventana en `grid.temporal_configs`:

```yaml
grid:
  temporal_configs:
    - id: ow600_lt10_pw100
      OW: 600
      LT: 10
      PW: 100
      window_strategy: synchro
      nan_mode: discard

    - id: ow600_lt100_pw100
      OW: 600
      LT: 100
      PW: 100
      window_strategy: synchro
      nan_mode: discard
```

Cada `id` debe ser ?nico. El orquestador cruza cada configuraci?n temporal con las medidas, direcciones y band sets.

La configuraci?n temporal forma parte del `job_key`. Por tanto, dos jobs con la misma medida y threshold pero distinto `OW`, `LT`, `PW`, `window_strategy` o `nan_mode` se ejecutan como jobs distintos.

## Bandas Y Effective Threshold

La comparaci?n se define con `event_band_sets`, no con un grid de thresholds. Cada band set tiene:

- `id`: nombre corto del conjunto de bandas.
- `values`: lista ordenada de porcentajes.

El `effective_threshold` se deriva as?:

- `high`: usa la ?ltima banda.
- `low`: usa la primera banda.

Ejemplo:

```yaml
event_band_sets:
  - id: coarse_10
    values: [10, 20, 30, 40, 50, 60, 70, 80, 90]
```

Con `direction: high`, el `effective_threshold` es `90`.

## Deduplicaci?n TS Vs Events/UniEvents

En `events` y `uniEvents`, el `job_key` incluye:

- configuraci?n temporal,
- medida,
- direcci?n,
- `bands_id`,
- lista completa de bandas,
- `effective_threshold`,
- configuraci?n de modelo.

Por eso `events` y `uniEvents` ejecutan un job por cada band set completo, aunque dos band sets tengan el mismo `effective_threshold`.

La diferencia entre ambos pipelines esta en F02:

- `events` genera F02 con bandas, strategy y `nan_mode`.
- `uniEvents` genera F02 con bandas, strategy, `nan_mode` y `MEASURE`.

En `ts`, el `job_key` incluye la configuraci?n temporal y el `effective_threshold`, pero no incluye `bands_id` ni la lista completa de bandas.

Por eso, si dos band sets tienen el mismo `effective_threshold`, TS se ejecuta una sola vez para esa medida, direcci?n y configuraci?n temporal.

## Ejecutar

Desde este repositorio:

```bash
python scripts/run_experiment.py
```

El script:

1. Lee `experiment.yaml`.
2. Expande los jobs para `ts`, `events` y `uniEvents`, segun `pipelines`.
3. Lee todos los manifests anteriores en `runs/*/manifest.csv`.
4. No repite jobs que ya est?n `COMPLETED`.
5. Crea una carpeta `runs/YYYY-MM-DD_NNN/` solo si hay jobs nuevos.
6. Ejecuta los jobs pendientes uno a uno.
7. Guarda logs, manifest y outputs configurados.

## Idempotencia

La identidad de cada job se calcula con un hash estable (`job_key`) a partir de su configuraci?n cient?fica.

Si un job est? `FAILED`, se puede reintentar en una ejecuci?n posterior. Solo se saltan los que ya est?n `COMPLETED`.

## Outputs

La secci?n `outputs:` de `experiment.yaml` controla qu? archivos se copian desde las variantes generadas.

Por defecto se excluyen artefactos pesados como:

- `.parquet`
- `.h5`
- `.tflite`
- `.pkl`
- `.npy`
- `.npz`

Tambi?n se puede ejecutar el collector sin lanzar experimentos:

```bash
python scripts/collect_outputs.py
```

## Global Model Index

Los manifests fechados en `runs/<run_id>/manifest.csv` son la trazabilidad de cada ejecucion. Para tener una vista global acumulada de todos los modelos creados, se puede regenerar:

```bash
python scripts/build_model_index.py
```

El script crea `registry/model_index.yaml`, un indice visual jerarquico agrupado por:

- medida,
- direccion,
- `effective_threshold`,
- configuracion temporal,
- bandas,
- pipeline.

Tambien incluye las variantes F02/F03/F04/F05, sus rutas reales dentro de los repos originales y la ruta local de outputs copiados en este repo comparador.

Para incluir filas fallidas en el indice de depuracion:

```bash
python scripts/build_model_index.py --include-failed
```

## Report

Se puede generar un report HTML simple a partir de `registry/model_index.yaml` y los outputs copiados en `runs/`:

```bash
python scripts/build_report.py
```

El resultado queda en:

```text
reports/report.html
```

El report agrupa los modelos por medida, direccion, `effective_threshold`, configuracion temporal, bandas y pipeline. Tambien incluye rankings top 10 por F1, recall y precision, y enlaza las variantes F02/F03/F04/F05 con sus rutas locales y originales cuando estan disponibles.

## Resultados

Cada run nuevo contiene:

- `runs/<run_id>/manifest.csv`: estado de cada job ejecutado en ese run.
- `runs/<run_id>/logs/`: stdout, stderr y comandos ejecutados.
- `runs/<run_id>/outputs/`: copia ligera de outputs configurados por fase.

Los outputs por job quedan en:

```text
runs/<run_id>/outputs/<pipeline>/<job_key>/f02/
runs/<run_id>/outputs/<pipeline>/<job_key>/f03/
runs/<run_id>/outputs/<pipeline>/<job_key>/f04/
runs/<run_id>/outputs/<pipeline>/<job_key>/f05/
```
