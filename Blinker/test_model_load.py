import dlib

try:
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
