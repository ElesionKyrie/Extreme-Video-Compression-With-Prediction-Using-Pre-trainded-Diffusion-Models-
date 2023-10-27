  
# Comment for all reviewers
  
We thank the reviewers for their time, effort, and suggestions. Based on the feedback, we have updated the paper by:
  
1. Including extra experimental details. In particular, the source code is released on GitHub, which can be found at: <https://github.com/ElesionKyrie/Extreme-Video-Compression-With-Prediction-Using-Pre-trainded-Diffusion-Models->. We have made some improvements to the code from our previous project. In the image compression process, we replaced BPG compression with a state-of-the-art neural network model, which has shown remarkable improvements. The open-source code is based on our improved model. The experimental results on the UVG dataset are also based on this enhanced model.
  
2. Extending our method by replacing the standard BPG compression with a state-of-the-art neural network model, which further improves the compression rates.
  
3. Comparing our method with neural network based baselines such as DCVC-DC. The results are shown below.
  
  
    | Methods \ Metrics | PSNR | LPIPS | FVD |
    | ------- | ------- | ------- | ------- |
    | H.265   | 22.44&plusmn;2.18  | 0.22&plusmn; 0.04| 3886.70&plusmn;1174.39   |
    | H.264   | 24.74&plusmn;2.48  | 0.13&plusmn;0.04  | 2414.48&plusmn; 830.35  |
    | Ours   | 23.70&plusmn;2.47   | 0.12&plusmn;0.03  | **613.93&plusmn;190.81**  |
    | DCVC-DC  | **34.68&plusmn;2.01**  | **0.037&plusmn;0.02**  | 745.86&plusmn; 401.16   |
  
  
  
  
  
3. Adding experiments to investigate the generalization ability of the proposed model. Specifically, we included additional tests on the UVG dataset. 
  
  
# Response to Comments from Reviewer 1
  
We thank the reviewer for their feedback and suggestions. Please see our common response above. 
  
# Response to Comments from Reviewer 2
  
We thank the reviewer for their detailed comments. 
  
> Consider both compression performance and computational efficiency, including parameter counts. I'm also curious about computation cost and memory consumption.
  
  
**Computation cost and memory consumption.** We employ the "Cityscapes concat" model from MCVD, which boasts a parameter count of 262.1M as documented by the authors. The total number of frames generated depends on a predefined threshold <img src="https://latex.codecogs.com/gif.latex?\rho"/>. In our experiments, each video consists of 30 frames at a resolution of <img src="https://latex.codecogs.com/gif.latex?128\times128"/>, and with a batch generation capability of 5 frames. We need a maximum of 5 generation cycles. The GPU memory consumption peaks at approximately 8.2GB with a batch size of 1, and in the worst-case scenario, the time required reaches 240 seconds.
  
> I am curious about how the generalization ability of this model is.     
  
  
>  While it uses the MNIST and Cityscape datasets, how about the performance in other dataset which are commonly used in traditional video compression methods, like UVG?
  
**Generalization ability.** In addition to the Cityscape dataset, we also evaluated the model's performance on the UVG dataset. From the results, we see that, even though the model was trained on the Cityscape dataset, it can still compress other videos. However, since the Cityscape dataset has relatively monotonous colors and small motion variations, when faced with highly diverse samples, the generative capability, and hence, the compression performance, may degrade.
  
In the table below, we present the compression results of our model on seven videos from the UVG dataset. We resized each video to <img src="https://latex.codecogs.com/gif.latex?128\times128"/> and selected the first <img src="https://latex.codecogs.com/gif.latex?30"/> frames. Additionally, we compared the compression performance with DCVC-DC (Neural Video Compression with Diverse Contexts, CVPR 2023). We updated the paper to include these results (also shown below)
  
  
  
|Bpp=0.06  | PSNR  | PSNR  | PSNR  | PSNR  | LPIPS  | LPIPS  | LPIPS  | LPIPS  | FVD | FVD | FVD | FVD |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Data&Method     | Ours     | H.264     | H.265     | DCVC-DC     | Ours     | H.264     | H.265     | DCVC-DC     | Ours     | H.264     | H.265     | DCVC-DC     |
| UVG-YachtRide     | 25.61     | 26.44     | 23.95     | **35.20**     | 0.096     | 0.075     | 0.14     | **0.009**     | 2540     | 1958     | 4282      | **218**     |
| UVG-Beauty     | 28.45     | **28.76**     | 25.28     | nan     | **0.057**     |  0.086     | 0.17     | nan     | **1416**     | 1913     | 3227     | nan     |
| UVG-Bosphorus     | 24.55     | **29.57**     | 26.70     | nan     |  0.101     | **0.053**    |  0.104     | nan     | **2079**     | 2275     | 2951     | nan     |
| UVG-HoneyBee     | nan     |  **25.37**     | 21.55     | nan     | nan     | **0.047**     | 0.196     | nan     | nan     | **554**     | 1446     | nan     |
| UVG-Jockey     | 22.82     | **22.91**     |  20.95     | nan     | 0.147     | **0.124**     | 0.201     | nan     | **2349**    | 4194     | 6426     | nan     |
| UVG-ReadySteadyGo     | 20.70     | 22.84     | 20.11     | **33.01**     | 0.112     | 0.155     | 0.316     | **0.032**     | 2832     | 3382     |  6347     | **902**     |
| UVG-ShakeNDry     | 24.68     | 26.59     | 24.43     | **36.68**     |  0.111     | 0.077     | 0.158     | 0.0257     | 1400     | 2126     | 2896     | **689**     |
| AverageValue | 24.47 | 30.41 | 23.28 | **34.96** | 0.104 | 0.088 | 0.184 | **0.022** | 2087 | 2343 | 4010 | **603** |
  
  
> The paper solely compares its method with H.264 and H.265, without exploring comparisons with other video compression methods.
  
**Comparison with other video compression methods.** On the Cityscape dataset, we compared our compression performance with that of DCVC-DC (Neural Video Compression with Diverse Contexts, CVPR 2023).
  
From the results, it can be seen that at bpp=0.06, our FVD metric achieved better performance than DCVC-DC.
  
  
| Methods&Metrics | PSNR | LPIPS | FVD |
| ------- | ------- | ------- | ------- |
| H.265   | 22.44&plusmn;2.18  | 0.22&plusmn; 0.04| 3886.70&plusmn;1174.39   |
| H.264   | 24.74&plusmn;2.48  | 0.13&plusmn;0.04  | 2414.48&plusmn; 830.35  |
| Ours   | 23.70&plusmn;2.47   | 0.12&plusmn;0.03  | **613.93&plusmn;190.81**  |
| DCVC-DC  | **34.68&plusmn;2.01**  | **0.037&plusmn;0.02**  | 745.86&plusmn; 401.16   |
  
  
>  There are other video prediction methods, how about their performance on video compression work compared to this method?
  
**Other video prediction methods in this method** The model we employed was SOTA on Cityscape dataset in video prediction task. Note that the compression performance of our model is related directly to the prediction capability of the underlying generative method. Hence, we did not consider other video prediction methods. 
  
  
> The PSNR of the proposed method is not as higher as that of H.264 for all BPP and of H.265 for larger BPP.
  
**PSNR results.** We would like to emphasize that the primary goal of the proposed method is to exploit the generative ability of the pre-trained neural network in the low bpp regimes. It has been shown that in such regimes, metrics such as LPIPS and FVD better capture the video reconstruction quality perceived by human users comparing to distortion measures such as PSNR. We will revise the discussion section to highlight this issue.
  
  
# Response to Comments from Reviewer 3
  
We thank the reviewer for the detailed comments.
  
> The quality threshold needs more discussion. How it is defined and how it affects the quality of compressed videos.
  
**Quality threshold $\rho$.** The definition of $\rho$ is given in Sec. 3.4 of our paper. It is a pre-defined distortion/perception threshold that controls whether a frame is encoded and transmitted, or generated at the receiver. The threshold depends on the requirements of the receiver and it directly determines the trade-off between the rate and reconstruction quality. 
In our experiments, we utilize the LPIPS metric, a widely recognized benchmark for assessing the perceptual quality of generated frames. The LPIPS metric between the reconstructed frame and the original frame is computed, and compared with the threshold $\rho$. We empirically test across the range of [0.02, 0.30] and obtain the envelope boundary of metrics for each video. As lower LPIPS score signifies superior quality, a smaller $\rho$. gives better overall reconstruction quality while increasing the required bpp.
  
  
  
> As the pre-trained diffusion model is used for video prediction (train and test on video prediction datasets), if the video topic changes, does the diffusion model have generalization ability. (For example, evaluating the current framework on UVG dataset.)
  
> If not, is the diffusion model highly relevant to the video content to be compressed? From my point of view, the performance of diffusion model is highly relied on training content. Hope the author can explain this.
  
  
**Generalization ability.** In addition to the Cityscape dataset, we also evaluated the model's generalization performance on the UVG dataset. From the results, even though the model was trained on the Cityscape dataset, it can still achieve a certain generation capability for out-of-distribution samples. However, it is natural that the performance will degrade as the video sequences become statistically less similar. For example, the samples in the Cityscape dataset have relatively monotonous colors and small motion variations. Hence, when faced with highly diverse datasets, the prediction capability of model is expected to decline. 
  
Please see our response above to a similar comment from Reviewer 2, where we provided results for some samples out of the Cityscape dataset.
  
  
> As the author only shows the results on video prediction datasets, I wonder if it is possible to train a diffusion model on a large video dataset like Vimeo-90k then evaluating on UVG and MCL-JC.
  
**Training diffusion models on larger dataset.** In the current work, we focus on applying **pre-trained** diffusion models to video compression instead of training new models. The primary goal of the current research is to apply the generative power of pre-trained neural networks to video compression tasks. As training the diffusion model on a dataset as large as Vimeo-90k is highly costly, we hope that such pre-trained models will be made available in the future for further extensions. We added these discussions to the conclusion section of the revised paper. 
  
  
> Lacking comparison results on different video compression methods (DVC and GAN-based methods) and datasets (UVG and MCL-JC).
  
  
**Comparation with other video compression method.** We have added a comparison  with DCVC-DC (Neural Video Compression with Diverse Contexts, CVPR 2023) on the Cityscape dataset. It can be seen that at bpp=0.06, our FVD metric achieved better performance than DCVC-DC.
  
| Methods&Metrics | PSNR | LPIPS | FVD |
| ------- | ------- | ------- | ------- |
| H.265   | 22.44&plusmn;2.18  | 0.22&plusmn; 0.04| 3886.70&plusmn;1174.39   |
| H.264   | 24.74&plusmn;2.48  | 0.13&plusmn;0.04  | 2414.48&plusmn; 830.35  |
| Ours   | 23.70&plusmn;2.47   | 0.12&plusmn;0.03  | **613.93&plusmn;190.81**  |
| DCVC-DC  | **34.68&plusmn;2.01**  | **0.037&plusmn;0.02**  | 745.86&plusmn; 401.16   |
  
  
