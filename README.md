# LiteFace - InsightFace with TensorFlow Lite
With LiteFace we convert the state-of-the-art face detection and recognition models InsightFace, from MXNet to TensorFlow Lite to be deployed and used in Android, iOS, embedded devices etc... for real-time face detection and recognition.  
There is no other documented way of doing this.

## Installation - Reference Important Notes Below
Clone and install everything we need:
```bash
git clone https://github.com/deepinsight/insightface.git
pip3 install mmdnn
cd insightface/python-package
pip3 install -e .
```

Download all InsightFace models, they should be in `~/.insightface/models/`:
```bash
python3 << END
from insightface.app import FaceAnalysis; FaceAnalysis()
END
```

Convert ArcFace face recognition model to TensorFlow.
```bash
mmconvert -sf mxnet -in ~/.insightface/models/arcface_r100_v1/model-symbol.json -iw ~/.insightface/models/arcface_r100_v1/model-0000.params -df tensorflow -om tf_arcface_100_v1 --inputShape 3,112,112 --dump_tag SERVING
```

Convert ArcFace TensorFlow model to TFLite:
```bash
tflite_convert \
  --output_file=tf_arcface_100_v1/tf_arcface_100_v1.tflite \
  --saved_model_dir=tf_arcface_100_v1
```

Since RetinaFace detection model has multiple outputs, you must modify the `MMDNN/conversion/tensorflow/saver.py` script like so:
![](https://i.gyazo.com/1aedf08ac5676c6bf379d8015e1042ca.png)

Now run MMDNN to convert RetinaFace:
```bash
mmconvert -sf mxnet -in ~/.insightface/models/retinaface_r50_v1/R50-symbol.json -iw ~/.insightface/models/retinaface_r50_v1/R50-0000.params -df tensorflow -om tf_retinaface_r50_v1 --inputShape 3,480,640 --dump_tag SERVING
```

Convert RetinaFace TensorFlow model to TFLite:
```bash
tflite_convert \
  --output_file=tf_retinaface_r50_v1/retinaface_r50_v1.tflite \
  --saved_model_dir=tf_retinaface_r50_v1
```


## Important Notes

### The appropriate MXnet and TensorFlow installations for your system are required (GPU/CPU, specific CUDA version).
----

### If you run into missing package errors e.g you miss `python-opencv` then run `pip3 install` for that package.
----
### If you run into:
```python
from .cv2 import *
ImportError: libSM.so.6: cannot open shared object file: No such file or directory
```
Install these packages:
```bash
apt-get install libsm6 libxext6 libxrender-dev
```
----
### If you run into:
```python
ValueError: Object arrays cannot be loaded when allow_pickle=False
```
That's because your numpy version is too recent where `allow_pickle=False` by default. Downgrade:
```bash
pip3 install numpy==1.16.1
```
----
