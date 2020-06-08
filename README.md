# You Can YOLO 

### iOS📱 + Python + Object Detection in Realtime

[![SwiftBadge](https://camo.githubusercontent.com/81b9c1ef24c359c78bef01ab308f002e18508000/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f53776966742d352e312d6f72616e6765)](https://camo.githubusercontent.com/81b9c1ef24c359c78bef01ab308f002e18508000/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f53776966742d352e312d6f72616e6765) [![XcodeBadge](https://camo.githubusercontent.com/09ed72f0fef2987a6ea9ddb10106cd2a14d87944/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f58636f64652d31312e332d626c7565)](https://camo.githubusercontent.com/09ed72f0fef2987a6ea9ddb10106cd2a14d87944/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f58636f64652d31312e332d626c7565) [![iOS](https://camo.githubusercontent.com/068f624eb1aea7290293a41532983b1519da346d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f694f532d31332e332d6c6967687467726579)](https://camo.githubusercontent.com/068f624eb1aea7290293a41532983b1519da346d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f694f532d31332e332d6c6967687467726579)![Generic badge](https://img.shields.io/badge/Python-3.5.2-green.svg)![Generic badge](https://img.shields.io/badge/Keras-2.2.4-red.svg)![Generic badge](https://img.shields.io/badge/Tensorflow-1.6.0-blue.svg)
----

## 📕 Result in iOS

<img width="1024" alt="스크린샷 2020-06-08 오후 9 38 30_t2" src="https://user-images.githubusercontent.com/46750574/84032022-a59cd800-a9d1-11ea-8d04-00431fa32086.png">

---

## 📕 Install Xcode

In order to develop for iOS we need to first install the latest version of Xcode, which can be found on the [Mac App Store](https://itunes.apple.com/us/app/xcode/id497799835?mt=12)

---

## 📕 Requirements

1. The test environment is
   - Python 3.5.2
   - Keras 2.2.4
   - tensorflow 1.6.0
2. Default anchors are used. If you use your own anchors, probably some changes are needed.
3. The inference result is not totally the same as Darknet but the difference is small.
4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.
5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.
6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.
7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.

---

## 📕 How to Make annotation coordinates in Python

* if you have mask image,
* Mask_offset("Mask_Image_path") in Python.
* **Of course You need to add Numpy**

```python
def mask_offset(mask) :
    min_x = np.min(np.where(mask == 0)[1])  # left top
    min_y = np.min(np.where(mask == 0)[0])  # left top
    max_x = np.max(np.where(mask == 0)[1])  # right bottom
    max_y = np.max(np.where(mask == 0)[0])  # right bottom 
    
    return min_x , min_y, max_x, max_y
```

> Example Mask Image

<img width="416" alt="스크린샷 2020-06-08 오후 9 17 25" src="https://user-images.githubusercontent.com/46750574/84029506-771cfe00-a9cd-11ea-93aa-b843356a3eda.png">


* If You Don't have mask images, Just Use ImageJ or  [LabelImg](https://github.com/tzutalin/labelImg).

### Custom DataSet Using Here [DataLink](https://public.roboflow.ai)

> Example image 

<img width="416" alt="스크린샷 2020-06-08 오후 9 18 40" src="https://user-images.githubusercontent.com/46750574/84030128-746ed880-a9ce-11ea-87e4-c7d560c6fa71.png">

> Example Annotations and Classes 

<img width="416" alt="스크린샷 2020-06-08 오후 9 18 29" src="https://user-images.githubusercontent.com/46750574/84029990-412c4980-a9ce-11ea-8b0b-0986e74b989a.png"><img width="416" alt="스크린샷 2020-06-08 오후 9 18 34" src="https://user-images.githubusercontent.com/46750574/84030019-4a1d1b00-a9ce-11ea-86ec-c2a04d386582.png">

---

## 📕 Prepare for Training 

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).

A Keras implementation of YOLOv4 (Tensorflow backend) inspired by [Ma-Dan/keras-yolo4](https://github.com/Ma-Dan/keras-yolo4).

### Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage

Use --help to see usage of yolo_video.py:

```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```

------

1. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

2. Generate your own annotation file and class names file.
   One row for one image;
   Row format: `image_file_path box1 box2 ... boxN`;
   Box format: `x_min,y_min,x_max,y_max,class_id` (no space).
   For VOC dataset, try `python voc_annotation.py`
   Here is an example:

   ```
   path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
   path/to/img2.jpg 120,300,250,600,2
   ...
   ```

3. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`
   The file model_data/yolo_weights.h5 is used to load pretrained weights.

4. Modify train.py and start training.
   `python train.py`
   Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:

1. wget https://pjreddie.com/media/files/darknet53.conv.74
2. rename it as darknet53.weights
3. python convert.py -w darknet53.cfg darknet53.weights mode_data/darknet53_weights.h5
4. use model_data/darknet53_weights.h5 in train.py

---

## 📕 Convert from MyYOLO to CoreML

First of all need to download YOLOv3 pretrained weights from [YOLO website](https://pjreddie.com/yolo/). Download both cfg and weights files.

Then load Darknet weights to Keras model using [Keras-YOLOv3](https://github.com/qqwweee/keras-yolo3) implementation.

After cloning above repo use this commend to load Darknet and save .h5:

```
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```

And finally to transform from .h5 keras model representation to CoreML format use code below:

```
import coremltools

coreml_model = coremltools.converters.keras.convert(
    'yolo.h5',
    input_names='image',
    image_input_names='image',
    input_name_shape_dict={'image': [None, 416, 416, 3]},
    image_scale=1/255.)

coreml_model.license = 'Public Domain'
coreml_model.input_description['image'] = 'Input image'

coreml_model.save('yolo.mlmodel')
```

---

## 📕 CoreML in Swift

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [YOLO-CoreML-MPSNNGraph](https://github.com/hollance/YOLO-CoreML-MPSNNGraph).

> Inputs : Image , Outputs : MultiArray(YOLOv3 has 3 MultiArrays , But Tiny-YOLOv3 has 2 MultiArrays.)

<img width="416" alt="스크린샷 2020-06-08 오후 9 38 30" src="https://user-images.githubusercontent.com/46750574/84031309-68841600-a9d0-11ea-91cf-7444d4edca02.png"><img width="416" alt="스크린샷 2020-06-08 오후 9 40 15" src="https://user-images.githubusercontent.com/46750574/84031454-a8e39400-a9d0-11ea-9e16-94755516dc4f.png">

### You only need to modify Something

> Helpers.swift

```swift
//YOLOv3 tiny
let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
//YOLOv3
let anchors: [Float] = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
//YOLOv4
let anchors: [Float] = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]

// The labels 
let labels = [
    "Your Data Labels",
]
```

>  YOLO.swift

```swift
    //YOLOv3 tiny
    let model = Tiny_YOLOv3()
    //YOLOv3
    let model = YOLOv3()
    //YOLOv4
    let model = yolov4()

    public init() { }    
    public func predict(image: CVPixelBuffer) -> [Prediction]? {
        
        if let output = try? model.prediction(image: image) {

            // Depending On Image Shapes..
            return computeBoundingBoxes(features: output._26)
            
        } else {
            return nil
        }
    }
    
    public func computeBoundingBoxes(features: MLMultiArray) -> [Prediction] {       
        // Depending On Image Shapes..
        assert(features.count == 18*26*26)
```

---

## 📕 Issue and Report

Please [**file**](https://github.com/chokyungjin/YouCanYOLO/issues) issues to send feedback or report a bug.

