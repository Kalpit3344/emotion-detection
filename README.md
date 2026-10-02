# Emotion Detection Project

A deep learning project for detecting emotions from facial images.

## Project Structure

```
emotion-detection/
├── data/
│   ├── train/          ← Training images organized by emotion
│   └── test/           ← Test images organized by emotion
├── haarcascades/
│   └── haarcascade_frontalface_default.xml (download required)
├── model/
│   ├── emotion_model.json     ← Model architecture (generated)
│   └── emotion_model.h5       ← Model weights (generated)
├── TrainEmotionDetector.py    ← Script to train the model
├── TestEmotionDetector.py     ← Script to test the model
└── requirements.txt           ← Project dependencies
```

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv emotion_env
   emotion_env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download haarcascade classifier:**
   - See `haarcascades/README.md` for download instructions

4. **Prepare your dataset:**
   - Organize training images in `data/train/` by emotion subdirectories
   - Organize test images in `data/test/` by emotion subdirectories
   - Expected emotions: angry, disgust, fear, happy, neutral, sad, surprise

## Usage

**Train the model:**
```bash
python TrainEmotionDetector.py
```

**Test the model:**
```bash
python TestEmotionDetector.py
```

## Emotion Categories

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

## Notes

- Ensure your dataset is properly organized with emotion subdirectories
- The model files will be generated in the `model/` folder after training
- Use the trained `emotion_model.h5` and `emotion_model.json` for predictions
