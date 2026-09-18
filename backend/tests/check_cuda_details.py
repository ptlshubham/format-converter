import os
import sys
import glob

# Add nvidia pip packages DLL directories
site_pkgs = os.path.join(sys.prefix, 'Lib', 'site-packages')
nvidia_dirs = glob.glob(os.path.join(site_pkgs, 'nvidia', '*', 'bin')) + glob.glob(os.path.join(site_pkgs, 'nvidia', '*', 'lib'))
for d in nvidia_dirs:
    if os.path.exists(d):
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
os.environ['PATH'] = ';'.join(nvidia_dirs) + ';' + os.environ.get('PATH', '')

import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

# Enable ORT logging to stdout to see exact CUDA loading logs
ort.set_default_logger_severity(0) # Verbose

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "segmentation", "birefnet-general", "model_fp16.onnx"))

print(f"Attempting to initialize CUDA session for {model_path}...")
try:
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("\nInferenceSession initialized!")
    print("Session active providers:", session.get_providers())
    active_p = session.get_providers()[0]
    if active_p == 'CUDAExecutionProvider':
        print("CUDA SUCCESS: Session actively using CUDAExecutionProvider!")
    else:
        print(f"CUDA SILENT FALLBACK DETECTED: Requested CUDA, but active provider is '{active_p}'")
except Exception as e:
    print(f"\nCUDA INFERENCE SESSION FAILED WITH EXCEPTION:\n{e}")
