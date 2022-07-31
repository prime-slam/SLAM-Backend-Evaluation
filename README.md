# SLAMBackendAlgos

A Project for benchmarking existing SLAM algorithms on planes dataset.
Benchmarking is done on an example of two TUM datasets:
office and living room with respect to given ground truth trajectories. An aim of the project is to evaluate ate and rpe
errors on both pre-annotated and frontend processed datasets, find key differences in the results and give minimum
requirements to the frontend algorithm.

# Example of usage
```python
python main.py C:\path_to_depth_images C:\path_to_color_images 1 0 1 3 C:\path_to_file_with_ground_truth
```
