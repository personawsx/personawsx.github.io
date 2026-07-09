Vedio=2D+Time（4D tensor：T×3×H×W）
Videos are ~30 frames per second (fps)
Size of uncompressed video
(3 bytes per pixel):
SD (640 x 480): ~1.5 GB per minute
HD (1920 x 1080): ~10 GB per minute
Solution: Train on short clips: low fps and low spatial resolution

<img width="985" height="503" alt="Image" src="https://github.com/user-attachments/assets/fc2bb18b-b6d2-4d03-9a63-45398adff8eb" /> 
# 1、Vedio Classification
## 1.1 Single-Frame CNN

<img width="981" height="492" alt="Image" src="https://github.com/user-attachments/assets/df32fdab-4864-4c43-8cd9-70f02a020017" />

## 1.2 Late Fusion
### a:（with FC layers）

<img width="992" height="492" alt="Image" src="https://github.com/user-attachments/assets/4882a5ea-8f18-4443-ad9c-923356a527a5" />

Problem：需要用到非常大的全连接层。

### b:（with pooling）

<img width="991" height="503" alt="Image" src="https://github.com/user-attachments/assets/2d0a2012-3042-4f0f-b4f3-f5485b4b0d57" />

Problem：可能会丢失某些信息。

## 1.2 Early Fusion
a：
<img width="986" height="422" alt="Image" src="https://github.com/user-attachments/assets/d9128f76-2692-4b77-9abc-5ccffd7a6e61" />

输入视频片段维度： $T×3×H×W$   
- $T$：帧数，每帧图像包含3通道RGB信息
张量重塑为： $3T×H×W$
- 将全部帧的RGB通道堆叠至通道维度，时序信息提前融合
Problem：仅依靠单一层卷积做时序处理，对长序列、复杂动作的建模能力不足

b：3D-CNN
<img width="988" height="415" alt="Image" src="https://github.com/user-attachments/assets/71616cc8-0a87-4027-8656-e0f51ccd8626" />

<img width="935" height="446" alt="Image" src="https://github.com/user-attachments/assets/26593b49-65ae-481b-8e59-abcfe8e93100" />

<img width="983" height="507" alt="Image" src="https://github.com/user-attachments/assets/77a4635f-6db8-4123-b481-2c76478ad49a" />

Late Fusion图示：
<img width="315" height="115" alt="Image" src="https://github.com/user-attachments/assets/1b49213a-1716-4ede-8c1a-c28b7446d1ef" />

## 1.3 3D卷积的图示

<img width="967" height="498" alt="Image" src="https://github.com/user-attachments/assets/667d6ff5-a4b0-4068-be39-fab5574f2396" />

## 1.4 C3D

<img width="967" height="503" alt="Image" src="https://github.com/user-attachments/assets/01213453-ded2-4885-8b78-3dff9c5b82b4" />

## 1.5

<img width="966" height="488" alt="Image" src="https://github.com/user-attachments/assets/dfd80ec7-ca17-45e1-9405-23775a0ce6d7" />

<img width="910" height="471" alt="Image" src="https://github.com/user-attachments/assets/00dda73e-45fd-4dc1-9829-f538171b927a" />