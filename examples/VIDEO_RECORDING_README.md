# Video Recording for SAC Model Evaluation

## Overview
The `sac_example.py` script has been modified to optionally record videos when running in evaluation mode (`train = False`). Video recording is controlled by feature flags, and you can evaluate either a single checkpoint or all available checkpoints.

## Feature Flags
- **`train`**: Set to `False` for evaluation mode, `True` for training mode
- **`record_video`**: Set to `True` to enable video recording, `False` to disable (only affects evaluation mode)
- **`evaluate_all_checkpoints`**: Set to `True` to evaluate all available checkpoints, `False` for single model evaluation

## Changes Made
1. **Added video recording imports**: `gymnasium.wrappers` and `time`
2. **Added feature flags**: Control video recording and multi-checkpoint evaluation
3. **Added `find_all_models()` function**: Discovers and sorts all available checkpoints
4. **Conditional render mode**: Uses `"rgb_array"` for video recording or `"human"` for live display
5. **Multi-checkpoint evaluation**: Iterates through all checkpoints when enabled
6. **Unique video names**: Each checkpoint gets its own video with timestep identifier
7. **Progress tracking**: Shows evaluation progress and results for each checkpoint
8. **Error handling**: Gracefully handles missing or corrupted model files

## Usage Modes

### 1. Single Checkpoint Evaluation (Default)
```python
train = False
record_video = True  # or False
evaluate_all_checkpoints = False
```
Evaluates a single model at the specified path.

### 2. All Checkpoints Evaluation
```python
train = False
record_video = True  # or False
evaluate_all_checkpoints = True
```
Evaluates all available checkpoints in the `models/` directory.

### 3. All Checkpoints without Video
```python
train = False
record_video = False
evaluate_all_checkpoints = True
```
Evaluates all checkpoints with live display (no video recording).

## Video Files (when record_video = True)

### Single Checkpoint Mode
- Video name: `sac_evaluation_<timestamp>`
- Location: `videos/` directory

### All Checkpoints Mode
- Video names: `sac_checkpoint_<timesteps>_<timestamp>`
- Examples: 
  - `sac_checkpoint_100000_1748619228-episode-0.mp4`
  - `sac_checkpoint_200000_1748619245-episode-0.mp4`
  - `sac_checkpoint_400000_1748619262-episode-0.mp4`
- Location: `videos/` directory

## Example Output

### All Checkpoints Evaluation
```
Found 5 checkpoints to evaluate

============================================================
Evaluating checkpoint 1/5
Model: models/sac_checkpoint_100000.zip
Timesteps: 100,000
============================================================
Recording video: sac_checkpoint_100000_1748619228
Checkpoint 100,000 results:
  - Total steps: 2847
  - Total reward: 1234.56
  - Average reward per step: 0.4338
  - Video saved: sac_checkpoint_100000_1748619228

============================================================
Evaluating checkpoint 2/5
Model: models/sac_checkpoint_200000.zip
Timesteps: 200,000
============================================================
...
```

## Configuration Options
- **Multi-checkpoint evaluation**: Toggle with `evaluate_all_checkpoints = True/False`
- **Video recording**: Toggle with `record_video = True/False`
- **Episode length limit**: Modify the `step_count > 5000` condition to change max episode length
- **Video directory**: Videos are saved in `videos/` directory (created automatically)
- **Models directory**: Checkpoints are searched in `models/` directory

## Behavior Matrix
| evaluate_all_checkpoints | record_video | Behavior |
|-------------------------|-------------|----------|
| False | True | Single model + video |
| False | False | Single model + live display |
| True | True | All checkpoints + videos |
| True | False | All checkpoints + live display |

## Use Cases
1. **Training Progress Visualization**: Use `evaluate_all_checkpoints = True` with `record_video = True` to create a video series showing learning progression
2. **Quick Single Test**: Use default settings for testing a specific checkpoint
3. **Live Debugging**: Use `record_video = False` for immediate visual feedback
4. **Batch Analysis**: Evaluate all checkpoints to find the best performing model

## Requirements
- The script requires `gymnasium` with video recording dependencies (only when `record_video = True`)
- Checkpoint models should be in the `models/` directory with naming pattern `sac_checkpoint_<timesteps>.zip`
- For single checkpoint evaluation, ensure the specified model path exists
