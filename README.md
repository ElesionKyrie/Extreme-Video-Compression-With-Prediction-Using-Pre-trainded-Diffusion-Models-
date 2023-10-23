# CPAL:Extreme Video Compression With Prediction Using Pre-trainded Diffusion Models 
##
For our project, the input is in the form of an array with a shape of (B, T, H, W, C), where each frame in the array has a fixed size of 128x128. The number of frames in each video is 30, resulting in a shape of (B, 30, 128, 128, 3). Before using this project, you may need to preprocess your video data accordingly.
You can control which videos to process by choosing the values for start_idx and end_idx. Ensure that the selected range does not exceed the value of B (the number of videos in your dataset).
```
python my_city_sender.py --data_npy "your npy path" --output_path "your path" --start_idx 0 --end_dix 1 
```
In the Benchmark section, we provide code for computing compression metrics for H.264 and H.265. The input for this code should be 30 frames of 128x128 image frames, preferably named in the format "frame%d."
```
python bench_uvg.py --dataset "your path" --outpath "your path" --project_srt uvg
```
##
Checkpoint

Regarding the checkpoints, we utilize two sets of them. One set includes "checkpoint_900000.pt," which is used for the video generation part. The other set contains six groups of checkpoints, and these checkpoints will be used for the image compression part, corresponding to six different compression qualities.
Checkpont coming soon
