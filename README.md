# mouse_project

## Installation Guide
1. Install the necessary dependencies: pyquaternion, scipy, numpy, pytorch, pyyaml, (open ai) gym, cython
2. Clone this respository
3. To install the mouse model libraries git clone the following repository into the top directory: 
    - [pybullet](https://github.com/bulletphysics/bullet3) 
    - [farms_pylog](https://gitlab.com/farmsim/farms_pylog.git)
    - [farms_container](https://gitlab.com/farmsim/farms_container.git)
    - [farms_muscle](https://gitlab.com/farmsim/farms_muscle.git)
4. Navigate to "/bullet3/examples/SharedMemory/SharedMemoryCommands.h"
    - Change the integers in lines(33-34) "#MAX_DEGREE_OF_FREEDOM 128" and "#MAX_NUM_SENSORS 256" to 512
5. Go to the root of the four cloned repos and type 
    ```
    pip install -e .
    ```

## Simulation
1. To begin training, enter the scripts folder and run:
    ```
    python main.py
    ``` 
2. To view a trained model, run:
    ```
    python main_trained.py
    ```