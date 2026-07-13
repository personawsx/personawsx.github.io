 
<img width="937" height="487" alt="Image" src="https://github.com/user-attachments/assets/01c7141a-26ef-4488-a8d2-52a9a34f9ea9" />
Pretext Objective：
能自动把视觉上同类的图片区分开、聚集在一起，但不能独立完成分类任务，也不认识任何类别名称

# 1、Self-Supervised pretext tasks
<img width="877" height="390" alt="Image" src="https://github.com/user-attachments/assets/5f5f8146-f110-440b-9a6d-30b5603a688b" />

### （1）主要任务：
<img width="883" height="442" alt="Image" src="https://github.com/user-attachments/assets/279cc097-9aa0-4a9f-8723-6e6fb989aeac" />

### （2）How to evaluate？
<img width="903" height="443" alt="Image" src="https://github.com/user-attachments/assets/1f616730-a7e9-49b6-a292-8cd8ca833fdd" />
其中，一般最关注的是下游任务的表现，即Transfer Learning and Downstream Task Performance

<img width="972" height="472" alt="Image" src="https://github.com/user-attachments/assets/964f1664-2fd7-4a8e-99d2-06aa65368650" />

自监督预训练：卷积层 conv + 全连接 fc，以预测旋转角度90°为训练目标，预训练完成后，会丢弃旋转预测用的全连接头，只保留卷积主干用于下游迁移
