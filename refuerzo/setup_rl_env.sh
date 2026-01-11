#!/bin/bash
set -e

echo "==============================================="
echo " INSTALANDO PYTHON 3.10.18 + VENV_RL (SEGURO)"
echo " SIN TOCAR PYTHON 3.8 DEL CONTENEDOR"
echo "==============================================="

# ----- 1. Instalar dependencias (NO afectan al Python del contenedor) -----
echo "[1/6] Instalando dependencias de compilación..."
sudo apt update
sudo apt install -y \
  build-essential \
  zlib1g-dev \
  libffi-dev \
  libssl-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  liblzma-dev \
  libncurses5-dev libncursesw5-dev \
  tk-dev \
  wget curl git

echo "Dependencias instaladas. Python 3.8 del contenedor NO ha sido modificado."
echo ""

# ----- 2. Instalar pyenv -----
echo "[2/6] Instalando pyenv (sin tocar Python del sistema)..."
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash
fi

# Añadir pyenv al bashrc si no está
if ! grep -q 'pyenv init' ~/.bashrc; then
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
fi

# Activar pyenv en esta sesión
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo "pyenv instalado."
echo ""

# ----- 3. Instalar Python 3.10.18 con pyenv -----
echo "[3/6] Instalando Python 3.10.18 (NO afecta al sistema)..."
pyenv install -s 3.10.18

PY310="$(pyenv prefix 3.10.18)/bin/python"
echo "Python 3.10 instalado en: $PY310"
echo ""

# ----- 4. Crear venv_rl -----
WORKDIR=~/ros_workspace/src/refuerzo
mkdir -p $WORKDIR
cd $WORKDIR

echo "[4/6] Creando venv_rl con Python 3.10.18..."
$PY310 -m venv venv_rl

echo "venv_rl creado en: $WORKDIR/venv_rl"
echo ""

# ----- 5. Instalar librerías dentro del venv -----
echo "[5/6] Instalando NumPy, Gymnasium, Stable-Baselines3 y SB3-contrib..."

source venv_rl/bin/activate
pip install --upgrade pip

pip install numpy==2.2.6
pip install "gymnasium==1.2.0"
pip install stable-baselines3[extra]==2.7.0
pip install sb3-contrib==2.7.0

deactivate

echo "Librerías instaladas correctamente."
echo ""

# ----- 6. Información final -----
echo "==============================================="
echo " ENTORNO venv_rl INSTALADO CON ÉXITO"
echo ""
echo " Usa este shebang en tu nodo refuerzo:"
echo "   #!$WORKDIR/venv_rl/bin/python"
echo ""
echo " Y en tu archivo .launch puedes añadir (opcional):"
echo "   <env name=\"PYTHONHOME\" value=\"$WORKDIR/venv_rl\" />"
echo "   <env name=\"PYTHONPATH\" value=\"$WORKDIR/venv_rl/lib/python3.10/site-packages\" />"
echo "   <env name=\"PATH\" value=\"$WORKDIR/venv_rl/bin:/usr/bin:/bin\" />"
echo ""
echo " Python 3.8 del contenedor sigue intacto."
echo " ROS seguirá usando Python 3.8 sin alteraciones."
echo " Solo tu nodo refuerzo usará Python 3.10.18 + NumPy/Gym/SB3 del venv."
echo "==============================================="
