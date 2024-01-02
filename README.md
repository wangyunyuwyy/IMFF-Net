# 1. Hardware specification<br>
Here is the hardware we used to produce the result

* CPU specs: Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20 GHz<br>
* Number of CPU cores: 20<br>
* GPU specs: NVIDIA Geforce RTX3080 GPU 10GB<br>
* Number of GPUs: 1 <br>
* Memory: 1TB<br>

# 2. OS and softwares<br>
* OS: Ubuntu 20.04 LTS<br>
* Cuda: 11.2<br> 
* Python: 3.8.0<br>
* Pytorch: 1.11.0

# 3. Dataset download<br> 
The dataset can be accessed from the following link：<br>
* https://drive.grand-challenge.org/Download/<br>
* https://cecas.clemson.edu/~ahoover/stare/<br>
* https://blogs.kingston.ac.uk/retinal/chasedb1/
  
# 4. Main contributions<br> 

This paper introduces a Multi-scale Feature Fusion segmentation network (IMFF-Net), which is built upon the foundation of a four-layer U-Shaped architecture. IMFF-Net leverages multi-scale feature fusion to enhance the utilization of both high-level and low-level features to improve segmentation performance. Experimental results demonstrate the superior performance of the proposed network over the current state-of-the-art networks. The primary contributions of this paper can be summarized as follows:<br>
* We present a retinal vessel segmentation network IMFF-Net distinguished by its robust multi-scale feature fusion capabilities. Specifically, IMFF-Net excels in the demanding task of segmenting retinal blood vessels, particularly in accurately capturing the intricate vessel structures present in retinal images. This effectiveness is achieved through the fusion of information across various scales.<br>
* To reduce spatial information loss stemming from multiple pooling operations, we propose the Attention Pooling Feature (APF) block. The APF block enables the network to effectively restrain noise levels within the feature map while preserving crucial features and emphasizing principal information.<br>
* To address challenges related to the accuracy of retinal microvessel segmentation, we propose the Upsampling and Downsampling Feature Fusion (UDFF) block. This block's objective is to facilitate multi-scale feature representation during both the downsampling and upsampling stages, thereby enabling a more comprehensive utilization of structural image features. This enhancement significantly improves the accuracy of microvessel segmentation.<br>


# 5. Result<br> 

* Result of proposed IMFF-Net on STARE dataset.<br>

Methods  | 𝑆𝑒  | 𝑆𝑝 | 𝐹1 | 𝐴𝑐𝑐 |
 ---- | ----- | ------ | ------| ------ 
IMFF-Net  | 0.8634 |	0.9869	| 0.8347	| 0.9707

* Result of proposed IMFF-Net on DRIVE dataset.<br>

Methods  | 𝑆𝑒  | 𝑆𝑝 | 𝐹1 | 𝐴𝑐𝑐 |
 ---- | ----- | ------ | ------| ------ 
IMFF-Net  | 0.8575 |	0.9860	| 0.7977	| 0.9621

* Result of proposed IMFF-Net on CHASE_DB1 dataset.<br>

Methods  | 𝑆𝑒  | 𝑆𝑝 | 𝐹1 | 𝐴𝑐𝑐 |
 ---- | ----- | ------ | ------| ------ 
IMFF-Net  | 0.8048	| 0.9867	| 0.7894	| 0.9730

# 6. Future issues<br> 
If you find any problems running the code, or have any questions regarding the solution, please contact me at: wangyunyu716@gmail.com and create an issue on the Repo's Issue tab.
