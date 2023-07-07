# Separate Maps

## Work Done

### The logic of the program

Similar to the algorihtm in separate maps, the program would take in a number of input arguments such as file name, number of threads, and cutoff, etc. and input its value to the `dictionary_builder()`, where the program uses a concurrent hashmap to process the file and resulting the 2-gram and 3-gram maps. It will spawn a number of threads based on the input where all of them work at the same time to process the file.

## Tech Details

### The Usage of `Dashmap`

For the concurrent map implementation, I used a popular third-party library called `dashmap`. Its implementation allows me to perform multi-threading without using locks and mutexes. Besides the change in data types, the code runs similarly compared to the separate map solution.

## Testing

I've pushed `test.sh` which covers the 9 given commands in `readme.md`. The details are explained below:

### Script

The script first checks out the initial commit, which is a sequential implementation of the program. It would then run the tests cases twice, once with the `single-map` flag and once without (although it does not matter in the original solution). Using `hyperfine`, it records the time it took (either in `ms` or `s`), and output the result into a folder. It then checkout to the current branch and perform the same action. In the end, it compares the time and result using the python script.

### Result (correctness & performance)

I've pushed the screenshot of the printed result, we can see that when the file is small, the performance did not get better (and they're even slightly worse) because when there's not much information to process, the program runs fast to the point that not all threads are used, however spawning threads requires us to `clone()` objects, which results in the loss of time compared to the sequential solution.

When the input files are large, however, multi-threading ensures much better performance, this is because when there're many lines to process, the threadpool would got filled and the processing is done concurrently.

The python script secures the correctness of the program.