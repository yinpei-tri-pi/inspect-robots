#!/usr/bin/env bash
# Install into a NEW local uv environment. No activation or system installation.
set -euo pipefail
DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBERO_SOURCE="${1:-/home/yinpei.dai/openpi/third_party/libero}"
DEMO_ENV="$DEMO_ROOT/.venv-libero"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/inspect-robots-uv-cache}"
export TMPDIR="${TMPDIR:-/dev/shm}"
export UV_LINK_MODE=copy
if [[ -e "$DEMO_ENV" ]]; then
  echo "Refusing to modify an existing environment: $DEMO_ENV" >&2
  exit 1
fi
test -d "$LIBERO_SOURCE/libero/libero/assets"
uv venv --python 3.11 "$DEMO_ENV"
uv pip install --python "$DEMO_ENV/bin/python" 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$DEMO_ENV/bin/python" 'numpy==1.26.4' 'mujoco==2.3.7' 'robosuite==1.4.1' 'numba==0.61.2' 'bddl==1.0.1' 'gym==0.25.2' scipy h5py matplotlib pillow imageio imageio-ffmpeg easydict cloudpickle pyyaml future -e "$DEMO_ROOT" -e "$DEMO_ROOT/plugins/inspect-robots-agent"
"$DEMO_ENV/bin/python" - "$LIBERO_SOURCE" "$DEMO_ROOT" <<'PY'
import json,site,sys
from pathlib import Path
source,root=map(Path,sys.argv[1:])
(Path(site.getsitepackages()[0])/'libero_source.pth').write_text(str(source.resolve())+'\n')
config=root/'artifacts/astra_libero/libero_config'
config.mkdir(parents=True,exist_ok=True)
base=source.resolve()/'libero/libero'
(config/'config.yaml').write_text(json.dumps({'benchmark_root':str(base),'bddl_files':str(base/'bddl_files'),'init_states':str(base/'init_files'),'datasets':str(base),'assets':str(base/'assets')},indent=2))
PY
# Ubuntu 22.04: download and extract matching Mesa libraries, without apt install.
NATIVE_STAGE="$(mktemp -d "$TMPDIR/inspect-robots-mesa.XXXXXX")"
pushd "$NATIVE_STAGE" >/dev/null
apt-get download libosmesa6 libglapi-mesa
for package in ./*.deb; do dpkg-deb -x "$package" extracted; done
mkdir -p "$DEMO_ENV/lib/native"
cp -a extracted/usr/lib/x86_64-linux-gnu/libOSMesa.so* extracted/usr/lib/x86_64-linux-gnu/libglapi.so* "$DEMO_ENV/lib/native/"
popd >/dev/null
uv pip freeze --python "$DEMO_ENV/bin/python" > "$DEMO_ROOT/artifacts/astra_libero/environment.freeze.txt"
echo "Installed isolated environment: $DEMO_ENV"
