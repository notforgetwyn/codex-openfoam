#!/bin/bash
cd /home/shihuayue/codex_project1
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
QT_QPA_PLATFORM=xcb \
QT_XCB_GL_INTEGRATION=xcb_egl \
PYTHONPATH=src \
python3 -X faulthandler -m foamdesk.app.main
