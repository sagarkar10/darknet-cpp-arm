# darknet-cpp-arm
Implementation of Darknet-Yolo in CPP for arm architecture
Tested in Pi3b running on Gentoo

### Dependencies:

* Arm Compute Library 
* Oprofile (optional, for profiling)

The setup uses Arm Compute Library and its Neon Support to immitate and optimize the performance of darknet-tiny-yolo on arm architecture.

### Arm Compute Library:

To build it the have sconscript.
Compile it natively or cross compile for arm-v8a (See the buid flags in the sconscripts by typing `scons --help`).
Here we use native compilation: `scons Werror=0 -j8 debug=0 asserts=0 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a build=native`
You can also cross-compile it with `scons Werror=0 -j8 debug=0 asserts=0 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a`

### Native Code:

The `makefile` is sufficient to build the code and generate a `darknet-cpp` binary which can be run with proper arguments (not req for now)
Look for inline documentation and block documentation for more info (wait for some time :p)

### Profile Code:

To profile the entire code we need to link the Arm Compute Library statically.
The `run.sh` script takes care of the profiling and genrating reports.

### TODO:

- [x] Get the Profiling to work on ArmCompute also
- [ ] Structure the code
- [ ] Document the code (so that it doesn't become another darknet :p)
- [ ] Complete Dependency on `.cfg` configfiles
- [ ] Command line arguments to make it more robust and test friendly
- [ ] Video Support
- [ ] WebCam support
- [ ] Profiling
- [ ] Detection is off by a constant location (have a look at `predictions.png`)
- [ ] Yolo-V2 support
- [ ] Consise the modules to a custom function (eg: `conv-activation-pool` can be a function that takes the req. params and do all of it )
- [ ] Unit Test Code

### GOTCHAS:(Any problem u face with this, probably I have faced it so here it goes)

* Try not to work in Gentoo
* Build oprofile from source. Follow simple steps : `./autoconf`, `./configure`, `make`, `make install` (dont do anything else in between after or before)
* Installing something in Gentoo (BRACE YOURSELF: ERROR is COMMING). Follow me : `sudo emerge --ask pkg_name`, Dont fear the errors (if any) do this `emerge --ask --autounmask-write pkg_name` and then press `use new / u` and then rerun the first command again `sudo emerge --ask pkg_name`
* Linking ARM Compute Statically :`--force-load {abs_path_to_ComputeLibrary}/ComputeLibrary/build/arm_compute/libarm_compute-static.a ../ComputeLibrary/build/arm_compute/libarm_compute_core-static.a -lpthread`
* Linking ARM Compute Dynamically :`L{abs_path_to_ComputeLibrary}/ComputeLibrary/build/arm_compute -larm_compute`
* Take care of `LD_LIBRARY_PATH` env-var, to set it `export LD_LIBRARY_PATH={abs_path_to_ComputeLibrary}/ComputeLibrary/build/arm_compute`
* Need `sudo` to run profiling.
* The `PATH` change on doing `sudo` so make sure your `operf` and `LD_LIBRARY_PATH` in still scope.
