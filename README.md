# traffic-sign-identification-with-AI

# Traffic Sign Recognition

This project includes an artificial intelligence model capable of recognizing traffic signs. A neural network model was created using TensorFlow and trained on a traffic sign dataset. The model can be easily tested through a GUI interface.

## Technologies Used
- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Tkinter (for GUI)
- Pandas

## Dataset
This project uses the [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset.

## File Structure
- `train.py`: Used for training the model.
- `test.py`: Tests the trained model for traffic sign prediction.
- `labels.csv`: Contains the traffic sign labels.
- `traffic_sign_model.keras`: The saved trained model.

## Setup
1. Install the required libraries:
   ```bash
   pip install tensorflow opencv-python pandas numpy
   ```

2. Organize your project files as follows:
   ```
   your_project/
   |-- train.py
   |-- test.py
   |-- labels.csv
   |-- traffic_sign_model.keras
   |-- trafikisaretleri/  # Dataset main folder
       |-- Train/
       |-- Test/
   ```

## Training
To train the model, follow these steps:

1. Update the file paths in `train.py` according to your system:
   - `labels_csv_path`: Path to the labels file
   - `inputBasePath`: Path to the dataset folder
   - `model_path`: Path where the model will be saved

2. Start training from the command line:
   ```bash
   python train.py
   ```

3. Once training is complete, the model will be saved as `traffic_sign_model.keras`.

## Testing
To test the model:

1. Verify the file paths in `test.py`:
   - `model_path`: Path to the trained model
   - `labels_csv_path`: Path to the labels file

2. Run the GUI for testing:
   ```bash
   python test.py
   ```

3. In the opened interface, select an image file to see the prediction result.

## Notes
- The model was trained on 90x90 images. The same dimensions are used for predictions.
- The model was trained using the `categorical_crossentropy` loss function and the `Adam` optimizer.

## Contribution
If you'd like to contribute to the project, please submit a pull request or report issues in the [Issues](#) section.

