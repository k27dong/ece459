# Separate Maps

Both the separate map & single map solution parallelize the looped call to `process_dictionary_builder_line()` as a producer-consumer model.

## Work Done

### The logic of the Program and the Change in `dictionary_builder()`

We know that the bottleneck for this problem lies in the `dictionary_builder()`, where the function reads all the lines of a file and then parse it using `process_dictionary_builder_line()` and update the content in the hashmap. To increase performance, I decide to parallelize the processing function and let multiple `process_dictionary_builder_line()` to run at the same time (detail will be explained in the next paragraph). The program begins from the `main` function which is barely modified, the only changed part is that now the program would pass `num_threads` and `single_map` into `parse_raw()` which allow us to use them in the builder, this would let the program to decide which algorithm to use (single map vs. concurrent map), and the number of thread to spawn in the threadpool. The parser would setup the hashmaps for double-maps and triple-maps, these two values would be further processed in the `main` function using the original logic given in the starter code. Those part remains sequential since their performance is acceptable.

## Tech Details

### File Read

The program would need to iterate through the entire log file, thus the first idea come to mind was to store the entire input in memory. However, this method is not efficient when dealing with a large log files, because the processing part only begin once the entire file is loaded into memory. Instead, I tried to process the Hashmaps during the iteration process. With the help of `threadpool`, for each iteration of the loop, I would call the processing function with the current line, the previous token and the lookahead line in a thread. When the iteration is finished, I wait until all the threads exit. By doing this, the performance of the program is maintained when the input file size is huge.

### Multi-threading with `ThreadPool`

Instead of using Rust's `thread` package in the standard library, I used `threadpool` for multi-threading the algorithm. This is because without knowing the number of jobs (number of lines needed to processed), I couldn't distribute the tasks to threads equally. `Threadpool` would balance the load of each thread: the jobs would be stored in a queue and they are automatically signed to thread that is available. Although `std::thread` can also achieve this but `threadpool` is a easier solution. When spawning the thread inside a loop, since the next iteration would come before the thread returns the ownership, I would have to `clone()` a copy of the value I need to pass to the thread (even if I know the value of this shared value would never change). This might slightly lower the performance but it's a necessary move to use thread safely. I have a vector of Hashmaps with length equal to the number of threads, where each thread would process on one of the Hashmaps. In the end, all hashmaps are joined together.

### Change in Function Declarations

Since I don't need `process_dictionary_builder_line()` to return the previous tokens, I've changed it into a void function which only update the passed in Hashmaps. I've made another function `find_prev_tokens` to do the job separately.

## Testing

For both implementations, I've tested performance using a script I wrote myself. The details would be explained in `concurrent-hashmaps.md`.