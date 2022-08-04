# SLAMBackendAlgos

A Project for benchmarking existing SLAM algorithms on planes dataset. 
Benchmarking is done on an example of two [ICL NUIM datasets](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html): 
office and living room with respect to given ground truth trajectories. An aim of the project is to evaluate ate and rpe
errors on both pre-annotated and frontend processed datasets, find key differences in the results and give minimum
requirements to the frontend algorithm.

# Example of usage
```python
Benchmarks a trajectory, built by an algorithm

positional arguments:
  main_data       Directory where main information files are stored
  annot           Directory where color images are stored
  {1,2,3}         living room = 1, office = 2, point clouds = 3
  first_node      From what node algorithm should start
  first_gt_node   From what node gt references start
  num_of_nodes    Number of needed nodes
  ds_filename_gt  Filename of a file with gt references

```
