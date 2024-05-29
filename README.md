# <center> ![logo](images/bir.jpg?raw=true)

# <center> YOLOv8 yordamida Face Detection

____



<p align="center">
  <img src="https://github.com/DilmurodGoyupov/Deploy-Face-Detection-YOLOv8-ONNXRuntime/blob/master/images/yolo.jpeg?raw=true" />
</p>


## <center> YOLOv8 ning nano modeli yordamida Face detection amalga oshiramiz

____

> 1-qadam. YOLO-ni o'rnatib olamiz. [Ultralytics](https://docs.ultralytics.com/ru)
```python
!pip install ultralytics
```
____
> 2-qadam. Quyidagi link orqali https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i/dataset/24/download/yolov8 datasetni yuklab olamiz
___
> 3-qadam. ***`YOLO Face Detection.ipynb` dagi kodlarni ishtalatmiz va modelni o'qitib olamiz.** *Agarda sizda `YOLO Face Detection.ipynb` ishlatib ko'rish imkoni bo'lmasa, quyidagi [link](https://drive.google.com/file/d/1fbYTZfsZUWQt7HfLnmbY4ieJX0N5R-0V/view?usp=sharing) orqali biz o'qitgan modelni saqlab oling*
___
> 4-qadam. YOLO modelimizni ONNX-ga o'tkazamiz olamiz.
```python
model.export(format="onnx")
```

<p align="center">
  <img src="https://github.com/DilmurodGoyupov/Deploy-Face-Detection-YOLOv8-ONNXRuntime/blob/master/images/onnx.png?raw=true" />
</p>


> 5-qadam. ONNXRuntime-ni o'rnatib olamiz.
```python
!pip install onnxruntime
```
___
> 6-qadam. `ONNX.ipynb` dagi kodlarni ishlatib ko'rib ONNXRuntime-ni qanday ishlatishni o'rganib olamiz. *Agarda sizda `ONNX.ipynb` ishlatib ko'rish imkoni bo'lmasa, quyidagi [link](https://drive.google.com/file/d/1xixqLwdSyw9DgLYwMp8WCMn1ZuYQCxIa/view?usp=sharing) orqali biz o'qitgan modelni saqlab oling*

___
> 7-qadam. `main.py` ni ishga tushiramiz.

+ 1-usul. Modelni kamera orqali ishga turshiramiz

```py
python main.py --video-path 0
```

+ 2-usul. Modelni videoda ishga tushiramiz

```
python main.py --video-path path/to/your/video.mp4
```



