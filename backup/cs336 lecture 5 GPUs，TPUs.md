<img width="1249" height="624" alt="Image" src="https://github.com/user-attachments/assets/b91361c4-65d6-4343-955c-d0a6ec56cfd4" />|

<img width="1200" height="660" alt="Image" src="https://github.com/user-attachments/assets/a850d887-6b50-4eae-bda4-312bd260cab6" />

<img width="1158" height="630" alt="Image" src="https://github.com/user-attachments/assets/ee356efa-21c3-4b7d-af07-8f2fe2fc83bb" />

早期算力增长依赖登纳德缩放（减少晶体管体积），而在该技术路线走到尽头后，依靠 GPU 并行算力的大幅提升，才支撑起现代大语言模型的持续迭代扩容。

<img width="1242" height="598" alt="Image" src="https://github.com/user-attachments/assets/edcc05e2-e4c7-4e65-b53c-a170aabfd4e4" />

CPU：优化低延迟、少量高速线程。
GPU：面向高吞吐、大规模并行线程，适合大规模矩阵运算。

<img width="1044" height="649" alt="Image" src="https://github.com/user-attachments/assets/bc733ff2-a40c-4d6a-ac16-1e6cb51fdb13" />

<img width="1734" height="982" alt="Image" src="https://github.com/user-attachments/assets/8c411b09-41af-47eb-96af-b6b6cabe47ea" />

<img width="1446" height="793" alt="Image" src="https://github.com/user-attachments/assets/9927cb3e-8ef7-4a3b-ba39-789460b6699a" />

硬件：GPU → 多个独立 SM（内含 L1 Cache、Shared Memory、SP、Tensor Core、Warp 调度器） + L2 Cache + Global Memory
软件：Thread ⊂ Warp ⊂ Block → 完整 Block 调度至单个 SM 运行

Block（线程块）：线程的分组，一个 Block 完整运行在同一个 SM 上，可使用该 SM 独享的 Shared Memory 共享内存。
Warp（线程束）：GPU 硬件实际调度执行的基本单位，固定由 32 个连续编号线程组成。
Thread（线程）：最小执行单元，并行完成计算。同一 Warp 内所有线程执行同一条指令，只是各自处理不同数据。

<img width="1394" height="798" alt="Image" src="https://github.com/user-attachments/assets/1b8731a5-c0ee-451d-9dca-fa9adeb939d7" />

<img width="1493" height="871" alt="Image" src="https://github.com/user-attachments/assets/d6550684-e906-4a68-a243-7753222f39b3" />

<img width="1555" height="800" alt="Image" src="https://github.com/user-attachments/assets/cc1d975a-baa0-4895-a6b3-6224312c6a9c" />

TPU中的Tensor Core相当于GPU中的SM，GPU中的Tensor Core是指矩阵乘法单元。

<img width="1113" height="851" alt="Image" src="https://github.com/user-attachments/assets/239411d0-ec58-4433-a7fe-7c3e695edaee" />

<img width="1442" height="772" alt="Image" src="https://github.com/user-attachments/assets/c9b8aa9f-79cf-4843-8b07-f4150358cafc" />

计算速度的增加远远超过内存传输的速度，因此现阶段大多优化是针对内存的。

<img width="1095" height="776" alt="Image" src="https://github.com/user-attachments/assets/dc33ab65-6452-4e68-8617-d4dfb558e336" />

最好是让点落在平稳期。

### How do we make GPUs go fast？