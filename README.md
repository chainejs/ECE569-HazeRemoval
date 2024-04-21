Dark Channel Haze Removal
=========================

MATLAB implementation of "[Single Image Haze Removal Using Dark Channel Prior][1]"


<img src="https://raw.githubusercontent.com/sjtrny/Dark-Channel-Haze-Removal/master/forest.jpg" width="200px"/>
&nbsp;
<img src="https://raw.githubusercontent.com/sjtrny/Dark-Channel-Haze-Removal/master/forest_recovered.jpg" width="200px"/>


CUDA acceleration implemented by Jarett Chaine, Evan Wiggins, and Gavin Mitchell.

Steps to compile and run our Haze Removal Algorithm:

1. Start Matlab GUI Session using the Interactive Apps located on the OnDemand Dashboard.

2. Once Matlab is open, navigate to the file within Matlab that contains the source code.

3. In the Matlab command prompt, compile the following four CUDA files one at a time:


mexcuda -v cuGetDarkChannel.cu

mexcuda -v cuGetAtmosphere.cu

mexcuda -v cuGetTransmission.cu

mexcuda -v getRadiance.cu

After each one is finished compiling you will see "MEX completed successfully"

4. set `useGPU = true` in demo_fast.m. (If set to False, the program will execute using a CPU only).

5. Run demo_fast.m - Successful completion should output a hazy image of a forest and a dehazed image of a forest. Execution time will also be an output on the command line.

6. If you want to compare results against the serial implementation, set `useGPU = false` in demo_fast.m, hit save, and re-run. Exection time is much slower.

7. To experiment with different hazy images simply replace forest.jpg on line 14 in demo_fast.m with one of the 10 other images available (haze1 - haze10). For example you could use haze1.jpg. 


