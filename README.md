```python
%cd alpr-unconstrained/
!ls
```

    /gdrive/My Drive/final_project/alpr/alpr-unconstrained
    annotation-tool.py  get-networks.sh		run.sh
    create-model.py     LICENSE			samples
    darknet		    license-plate-detection.py	src
    data		    license-plate-ocr.py	train-detector.py
    gen-outputs.py	    README.md			vehicle-detection.py



```python
%cd darknet
!make
```


```python
!/bin/bash get-networks.sh
```


```python
!/bin/bash run.sh -i samples/test -o /tmp/output -c /tmp/output/results.csv
```

    layer     filters    size              input                output
        0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32  0.299 BFLOPs
        1 max          2 x 2 / 2   416 x 416 x  32   ->   208 x 208 x  32
        2 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64  1.595 BFLOPs
        3 max          2 x 2 / 2   208 x 208 x  64   ->   104 x 104 x  64
        4 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128  1.595 BFLOPs
        5 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64  0.177 BFLOPs
        6 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128  1.595 BFLOPs
        7 max          2 x 2 / 2   104 x 104 x 128   ->    52 x  52 x 128
        8 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
        9 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
       10 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
       11 max          2 x 2 / 2    52 x  52 x 256   ->    26 x  26 x 256
       12 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
       13 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
       14 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
       15 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
       16 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
       17 max          2 x 2 / 2    26 x  26 x 512   ->    13 x  13 x 512
       18 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
       19 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
       20 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
       21 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
       22 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
       23 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024  3.190 BFLOPs
       24 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024  3.190 BFLOPs
       25 route  16
       26 conv     64  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x  64  0.044 BFLOPs
       27 reorg              / 2    26 x  26 x  64   ->    13 x  13 x 256
       28 route  27 24
       29 conv   1024  3 x 3 / 1    13 x  13 x1280   ->    13 x  13 x1024  3.987 BFLOPs
       30 conv    125  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 125  0.043 BFLOPs
       31 detection
    mask_scale: Using default '1.000000'
    Loading weights from data/vehicle-detector/yolo-voc.weights...Done!
    Searching for vehicles using YOLO...
    	Scanning samples/test/03009.jpg
    		0 cars found
    	Scanning samples/test/03016.jpg
    		0 cars found
    	Scanning samples/test/03025.jpg
    		0 cars found
    	Scanning samples/test/03033.jpg
    		0 cars found
    	Scanning samples/test/03057.jpg
    		0 cars found
    	Scanning samples/test/03058.jpg
    		0 cars found
    	Scanning samples/test/03066.jpg
    		0 cars found
    	Scanning samples/test/03071.jpg
    		0 cars found
    Using TensorFlow backend.
    2020-05-08 17:12:40.997155: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 17:12:42.977847: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2020-05-08 17:12:42.997558: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:12:42.998527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
    coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
    2020-05-08 17:12:42.998580: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 17:12:43.000624: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-08 17:12:43.004251: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-08 17:12:43.004643: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-08 17:12:43.007066: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-08 17:12:43.008126: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-08 17:12:43.012822: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 17:12:43.012989: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:12:43.013808: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:12:43.014548: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-08 17:12:43.022016: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2200000000 Hz
    2020-05-08 17:12:43.022271: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x23a2bc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-05-08 17:12:43.022310: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-05-08 17:12:43.078752: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:12:43.079704: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x23a2d80 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-05-08 17:12:43.079753: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
    2020-05-08 17:12:43.080062: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:12:43.080775: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
    coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
    2020-05-08 17:12:43.080835: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 17:12:43.080909: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-08 17:12:43.080957: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-08 17:12:43.080995: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-08 17:12:43.081035: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-08 17:12:43.081075: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-08 17:12:43.081115: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 17:12:43.081204: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:12:43.081978: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:12:43.082640: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-08 17:12:43.082699: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 17:12:43.502392: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-05-08 17:12:43.502474: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 
    2020-05-08 17:12:43.502504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N 
    2020-05-08 17:12:43.502737: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:12:43.503581: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:12:43.504415: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-05-08 17:12:43.504476: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10634 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
    Searching for license plates using WPOD-NET
    layer     filters    size              input                output
        0 conv     32  3 x 3 / 1   240 x  80 x   3   ->   240 x  80 x  32  0.033 BFLOPs
        1 max          2 x 2 / 2   240 x  80 x  32   ->   120 x  40 x  32
        2 conv     64  3 x 3 / 1   120 x  40 x  32   ->   120 x  40 x  64  0.177 BFLOPs
        3 max          2 x 2 / 2   120 x  40 x  64   ->    60 x  20 x  64
        4 conv    128  3 x 3 / 1    60 x  20 x  64   ->    60 x  20 x 128  0.177 BFLOPs
        5 conv     64  1 x 1 / 1    60 x  20 x 128   ->    60 x  20 x  64  0.020 BFLOPs
        6 conv    128  3 x 3 / 1    60 x  20 x  64   ->    60 x  20 x 128  0.177 BFLOPs
        7 max          2 x 2 / 2    60 x  20 x 128   ->    30 x  10 x 128
        8 conv    256  3 x 3 / 1    30 x  10 x 128   ->    30 x  10 x 256  0.177 BFLOPs
        9 conv    128  1 x 1 / 1    30 x  10 x 256   ->    30 x  10 x 128  0.020 BFLOPs
       10 conv    256  3 x 3 / 1    30 x  10 x 128   ->    30 x  10 x 256  0.177 BFLOPs
       11 conv    512  3 x 3 / 1    30 x  10 x 256   ->    30 x  10 x 512  0.708 BFLOPs
       12 conv    256  3 x 3 / 1    30 x  10 x 512   ->    30 x  10 x 256  0.708 BFLOPs
       13 conv    512  3 x 3 / 1    30 x  10 x 256   ->    30 x  10 x 512  0.708 BFLOPs
       14 conv     80  1 x 1 / 1    30 x  10 x 512   ->    30 x  10 x  80  0.025 BFLOPs
       15 detection
    mask_scale: Using default '1.000000'
    Loading weights from data/ocr/ocr-net.weights...Done!
    Performing OCR...
    rm: cannot remove '/tmp/output/*_lp.png': No such file or directory



```python
!mkdir models
```


```python
# create model
!python3 create-model.py eccv models/eccv-model-scracth
```

    Using TensorFlow backend.
    2020-05-08 17:16:32.216938: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    Creating model eccv
    2020-05-08 17:16:34.187942: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2020-05-08 17:16:34.207398: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:16:34.208211: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
    coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
    2020-05-08 17:16:34.208261: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 17:16:34.210300: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-08 17:16:34.213820: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-08 17:16:34.214183: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-08 17:16:34.216289: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-08 17:16:34.225844: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-08 17:16:34.231382: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 17:16:34.231535: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:16:34.232347: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:16:34.233154: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-08 17:16:34.240052: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2200000000 Hz
    2020-05-08 17:16:34.241176: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1edabc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-05-08 17:16:34.241217: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-05-08 17:16:34.298830: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:16:34.299740: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1edad80 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-05-08 17:16:34.299776: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
    2020-05-08 17:16:34.300029: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:16:34.300726: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
    coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
    2020-05-08 17:16:34.300784: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 17:16:34.300839: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-08 17:16:34.300898: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-08 17:16:34.300943: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-08 17:16:34.301001: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-08 17:16:34.301040: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-08 17:16:34.301079: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 17:16:34.301169: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:16:34.301943: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:16:34.302705: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-08 17:16:34.302762: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 17:16:34.735780: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-05-08 17:16:34.735838: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 
    2020-05-08 17:16:34.735858: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N 
    2020-05-08 17:16:34.736091: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:16:34.736973: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 17:16:34.737703: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-05-08 17:16:34.737760: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10634 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
    Finished
    Saving at models/eccv-model-scracth



```python
# train
!python3 train-detector.py --model models/eccv-model-scracth.h5 \
                           --name my-trained-model \
                           --train-dir samples/train-detector \
                           --output-dir models/my-trained-model/ \
                           -op Adam -lr .001 -its 300000 -bs 32 -its 30
```

    Using TensorFlow backend.
    2020-05-08 18:18:50.845121: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 18:18:52.943533: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2020-05-08 18:18:52.962090: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:18:52.962924: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
    coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
    2020-05-08 18:18:52.962976: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 18:18:52.965376: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-08 18:18:52.967473: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-08 18:18:52.967896: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-08 18:18:52.970303: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-08 18:18:52.971564: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-08 18:18:52.976541: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 18:18:52.976702: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:18:52.977544: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:18:52.978265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-08 18:18:52.984630: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2200000000 Hz
    2020-05-08 18:18:52.984900: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x32b4bc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-05-08 18:18:52.984936: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-05-08 18:18:53.041028: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:18:53.041976: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x32b4d80 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-05-08 18:18:53.042018: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
    2020-05-08 18:18:53.042196: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:18:53.042906: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
    coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
    2020-05-08 18:18:53.042976: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 18:18:53.043057: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-08 18:18:53.043113: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-08 18:18:53.043151: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-08 18:18:53.043187: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-08 18:18:53.043221: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-08 18:18:53.043259: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 18:18:53.043357: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:18:53.044087: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:18:53.044770: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-08 18:18:53.044827: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 18:18:53.489195: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-05-08 18:18:53.489260: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 
    2020-05-08 18:18:53.489281: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N 
    2020-05-08 18:18:53.489539: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:18:53.490375: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:18:53.491171: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-05-08 18:18:53.491237: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10634 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
    Checking input directory...
    3 images with labels found
    Iter. 1 (of 30)
    2020-05-08 18:20:05.906748: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 18:20:10.162283: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    	Loss: 397.409088
    Iter. 2 (of 30)
    	Loss: 459.891327
    Iter. 3 (of 30)
    	Loss: 284.081482
    Iter. 4 (of 30)
    	Loss: 151.177551
    Iter. 5 (of 30)
    	Loss: 133.835266
    Iter. 6 (of 30)
    	Loss: 97.503761
    Iter. 7 (of 30)
    	Loss: 100.953773
    Iter. 8 (of 30)
    	Loss: 75.872498
    Iter. 9 (of 30)
    	Loss: 76.575623
    Iter. 10 (of 30)
    	Loss: 58.591148
    Iter. 11 (of 30)
    	Loss: 76.668594
    Iter. 12 (of 30)
    	Loss: 44.083961
    Iter. 13 (of 30)
    	Loss: 49.745728
    Iter. 14 (of 30)
    	Loss: 57.257080
    Iter. 15 (of 30)
    	Loss: 45.728245
    Iter. 16 (of 30)
    	Loss: 30.480156
    Iter. 17 (of 30)
    	Loss: 44.242668
    Iter. 18 (of 30)
    	Loss: 55.801147
    Iter. 19 (of 30)
    	Loss: 37.023888
    Iter. 20 (of 30)
    	Loss: 43.306618
    Iter. 21 (of 30)
    	Loss: 38.562145
    Iter. 22 (of 30)
    	Loss: 35.364933
    Iter. 23 (of 30)
    	Loss: 38.357830
    Iter. 24 (of 30)
    	Loss: 33.510368
    Iter. 25 (of 30)
    	Loss: 28.245598
    Iter. 26 (of 30)
    	Loss: 28.770166
    Iter. 27 (of 30)
    	Loss: 35.092434
    Iter. 28 (of 30)
    	Loss: 33.335228
    Iter. 29 (of 30)
    	Loss: 28.580080
    Iter. 30 (of 30)
    	Loss: 34.525200
    Stopping data generator
    Saving model (models/my-trained-model//my-trained-model_final)



```python
# OCR
!python3 license-plate-detection.py samples/test data/lp-detector/wpod-net_update1.h5
```

    Using TensorFlow backend.
    2020-05-08 18:36:23.084884: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 18:36:25.111793: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2020-05-08 18:36:25.136416: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:36:25.137283: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
    coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
    2020-05-08 18:36:25.137327: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 18:36:25.139673: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-08 18:36:25.141688: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-08 18:36:25.142206: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-08 18:36:25.144451: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-08 18:36:25.145719: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-08 18:36:25.150483: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 18:36:25.150602: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:36:25.151346: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:36:25.152034: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-08 18:36:25.158432: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2200000000 Hz
    2020-05-08 18:36:25.158688: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x14d6bc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-05-08 18:36:25.158727: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-05-08 18:36:25.216040: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:36:25.216996: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x14d6d80 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-05-08 18:36:25.217031: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
    2020-05-08 18:36:25.217338: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:36:25.218074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
    coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
    2020-05-08 18:36:25.218142: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 18:36:25.218201: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-05-08 18:36:25.218241: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-05-08 18:36:25.218301: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-05-08 18:36:25.218360: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-05-08 18:36:25.218400: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-05-08 18:36:25.218451: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 18:36:25.218539: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:36:25.219413: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:36:25.220182: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
    2020-05-08 18:36:25.220241: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-05-08 18:36:25.641728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-05-08 18:36:25.641807: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 
    2020-05-08 18:36:25.641825: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N 
    2020-05-08 18:36:25.642122: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:36:25.642944: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-05-08 18:36:25.643652: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-05-08 18:36:25.643707: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10634 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
    Searching for license plates using WPOD-NET
    	 Processing samples/test/03009.jpg
    		Bound dim: 384, ratio: 1.333333
    2020-05-08 18:36:28.531886: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-05-08 18:36:29.412887: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    	 Processing samples/test/03016.jpg
    		Bound dim: 432, ratio: 1.503006
    	 Processing samples/test/03025.jpg
    		Bound dim: 478, ratio: 1.637405
    	 Processing samples/test/03033.jpg
    		Bound dim: 384, ratio: 1.333333
    	 Processing samples/test/03057.jpg
    		Bound dim: 446, ratio: 1.499268
    	 Processing samples/test/03058.jpg
    		Bound dim: 384, ratio: 1.333333
    	 Processing samples/test/03066.jpg
    		Bound dim: 384, ratio: 1.333333
    	 Processing samples/test/03071.jpg
    		Bound dim: 472, ratio: 1.600000



```python
# OCR
!python3 license-plate-ocr.py samples/test/
```

    layer     filters    size              input                output
        0 conv     32  3 x 3 / 1   240 x  80 x   3   ->   240 x  80 x  32  0.033 BFLOPs
        1 max          2 x 2 / 2   240 x  80 x  32   ->   120 x  40 x  32
        2 conv     64  3 x 3 / 1   120 x  40 x  32   ->   120 x  40 x  64  0.177 BFLOPs
        3 max          2 x 2 / 2   120 x  40 x  64   ->    60 x  20 x  64
        4 conv    128  3 x 3 / 1    60 x  20 x  64   ->    60 x  20 x 128  0.177 BFLOPs
        5 conv     64  1 x 1 / 1    60 x  20 x 128   ->    60 x  20 x  64  0.020 BFLOPs
        6 conv    128  3 x 3 / 1    60 x  20 x  64   ->    60 x  20 x 128  0.177 BFLOPs
        7 max          2 x 2 / 2    60 x  20 x 128   ->    30 x  10 x 128
        8 conv    256  3 x 3 / 1    30 x  10 x 128   ->    30 x  10 x 256  0.177 BFLOPs
        9 conv    128  1 x 1 / 1    30 x  10 x 256   ->    30 x  10 x 128  0.020 BFLOPs
       10 conv    256  3 x 3 / 1    30 x  10 x 128   ->    30 x  10 x 256  0.177 BFLOPs
       11 conv    512  3 x 3 / 1    30 x  10 x 256   ->    30 x  10 x 512  0.708 BFLOPs
       12 conv    256  3 x 3 / 1    30 x  10 x 512   ->    30 x  10 x 256  0.708 BFLOPs
       13 conv    512  3 x 3 / 1    30 x  10 x 256   ->    30 x  10 x 512  0.708 BFLOPs
       14 conv     80  1 x 1 / 1    30 x  10 x 512   ->    30 x  10 x  80  0.025 BFLOPs
       15 detection
    mask_scale: Using default '1.000000'
    Loading weights from data/ocr/ocr-net.weights...Done!
    Performing OCR...
    	Scanning samples/test/03009.jpg
    No characters found
    	Scanning samples/test/03016.jpg
    No characters found
    	Scanning samples/test/03025.jpg
    No characters found
    	Scanning samples/test/03033.jpg
    No characters found
    	Scanning samples/test/03057.jpg
    No characters found
    	Scanning samples/test/03058.jpg
    No characters found
    	Scanning samples/test/03066.jpg
    		LP: H
    	Scanning samples/test/03071.jpg
    No characters found

