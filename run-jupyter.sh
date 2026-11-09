#!/bin/bash

# --- CONFIGURACIÓN ---
# Ruta absoluta a tu entorno virtual (basado en tus logs anteriores)
ENV_PATH="/home/moybadajoz/Documents/GaitTrakingProyect/.venv-jupyter"
PORT=8888

# --- 1. ACTIVAR ENTORNO ---
echo "🔌 Activando entorno virtual en: $ENV_PATH..."
source "$ENV_PATH/bin/activate"

# --- 2. FORZAR BACKEND DE PYTORCH (Solución RTX 5070) ---
# Esto evita que tengas que poner os.environ en cada notebook
export KERAS_BACKEND="torch"
echo "✅ Backend configurado a: PYTORCH (para compatibilidad con RTX 5070)"

# --- 3. INICIAR JUPYTER PARA COLAB ---
echo "🚀 Iniciando Jupyter Notebook..."
echo "⚠️  Copia la URL con el token que aparecerá abajo y pégala en Colab."
echo "------------------------------------------------------------------"

# Ejecuta Jupyter con los permisos para Google Colab y sin reintentos de puerto
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=$PORT \
  --NotebookApp.port_retries=0