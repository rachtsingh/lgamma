# lgamma
Implementations of polygamma, lgamma, and beta functions for PyTorch. It's very hacky, but that's usually ok for research use.

To build, run:
    
    ./make.sh

You'll probably need to pass in the correct CUDA path to ```build.py```, which is run inside ```make.sh```, so modify it to instead call

    python build.py --cuda-path YOUR_CUDA_PATH
    
Also, you'll probably need to change the architecture version/CUDA compute capability inside ```make.sh```, so replace ```sm_35``` with whatever your [GPU supports](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
Feel free to open an issue if you run into another problem!
