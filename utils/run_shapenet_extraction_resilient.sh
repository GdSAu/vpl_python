#!/bin/bash
# Relanza extract_nbv_dataset.py hasta que termine con exito (exit 0).
# NBVDatasetExtractor.run() ya salta objetos que ya tienen su carpeta
# Point_cloud/NBVDE hecha, asi que cada reintento retoma donde quedo el anterior.

set -u
CONFIG="${1:-dataexample/paramsNBVDE_shapenet.yaml}"
MAX_RETRIES=100
LOGDIR="logs_shapenet_extraction"
mkdir -p "$LOGDIR"

source /home/sknkllr/miniconda3/etc/profile.d/conda.sh
conda activate o3d
export LD_LIBRARY_PATH="/home/sknkllr/miniconda3/envs/o3d/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
cd /mnt/6C24E28478939C77/Saulo/vpl_python

attempt=1
while [ "$attempt" -le "$MAX_RETRIES" ]; do
    ts=$(date +"%Y%m%d_%H%M%S")
    logfile="$LOGDIR/attempt_${attempt}_${ts}.log"
    echo "=== intento $attempt: $(date) ===" | tee -a "$LOGDIR/resumen.log"

    python extract_nbv_dataset.py "$CONFIG" > "$logfile" 2>&1
    code=$?

    if [ "$code" -eq 0 ]; then
        echo "=== terminado OK en intento $attempt: $(date) ===" | tee -a "$LOGDIR/resumen.log"
        exit 0
    fi

    echo "=== intento $attempt fallo (exit $code), ver $logfile ===" | tee -a "$LOGDIR/resumen.log"
    attempt=$((attempt + 1))
    sleep 10
done

echo "=== se alcanzo el limite de $MAX_RETRIES reintentos sin terminar ===" | tee -a "$LOGDIR/resumen.log"
exit 1
