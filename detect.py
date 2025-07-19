import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow import keras
from scipy.spatial import distance
from train import preprocess_face


class FaceEmbedder:
	def __init__(self, model_path):
		self.model = keras.models.load_model(model_path, compile=False)
	
	def embed_image(self, image):
		if isinstance(image, str):
			img = cv2.imread(image)
		else:
			img = image
		img = preprocess_face(img)
		tensor = np.expand_dims(img, 0)
		return self.model({'input_layer': tensor}, training=False).numpy().squeeze()
     

class FaceDB:
    def __init__(self, similarity_threshold=0.7, images_dir='images/registered'):
        self.embeddings = []
        self.labels = []
        self.similarity_threshold = similarity_threshold
        self.images_dir = images_dir
        self.face_embedder = FaceEmbedder('models/metric_embedding.keras')
        self.load_registered_faces()
    
    def load_registered_faces(self):
        for img_file in Path(self.images_dir).glob('*.jpg'):
            person_name = img_file.stem
            embedding = self.face_embedder.embed_image(str(img_file))
            self.embeddings.append(embedding)
            self.labels.append(person_name)
    
    def find_match(self, query_embedding):
        if not self.embeddings:
            return 'unknown', 0.0
        similarities = [1 - distance.cosine(query_embedding, emb) for emb in self.embeddings]
        best_idx = int(np.argmax(similarities))
        best_similarity = similarities[best_idx]
        best_match = self.labels[best_idx] if best_similarity >= self.similarity_threshold else 'unknown'
        return best_match, best_similarity
    
    def get_stats(self):
        return {'total_people': len(self.labels), 'threshold': self.similarity_threshold}


class LivenessDetector:
    def __init__(self, model_path='models/anti_spoofing.tflite', threshold=0.5):
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp  = self.interp.get_input_details()[0]
        self.out  = self.interp.get_output_details()[0]
        _, self.h, self.w, _ = self.inp['shape']
        self.threshold = threshold

    def is_live(self, face):
        img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.w, self.h)).astype('float32') / 255.0
        img = np.expand_dims(img, 0)
        self.interp.set_tensor(self.inp['index'], img)
        self.interp.invoke()
        logits = self.interp.get_tensor(self.out['index']).flatten()
        spoof_probs = 1.0 / (1.0 + np.exp(-logits))
        live_score  = float((1.0 - spoof_probs).mean())
        return live_score > self.threshold, live_score


class EmotionDetector:
    def __init__(self, model_path='models/emotion_detection.hdf5'):
        self.model = keras.models.load_model(model_path, compile=False)
        _, self.h, self.w, self.c = self.model.input_shape
        self.labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def predict(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (self.w, self.h))
        face = face.astype('float32') / 255.
        face = np.expand_dims(face, -1)
        face = np.expand_dims(face, 0)
        pred = int(np.argmax(self.model.predict(face, verbose=0)))
        return self.labels[pred]


class FaceRecognitionSystem:
    def __init__(self, embed_model_path):
        self.face_embedder = FaceEmbedder(embed_model_path)
        self.liveness_detector = LivenessDetector()
        self.emotion_detector = EmotionDetector()
        self.face_db = FaceDB()
    
    def extract_face_embedding(self, face_image):
        return self.face_embedder.embed_image(face_image)
    
    def process_frame(self, frame):
        annotated_frame = frame.copy()
        is_live, liveness_score = self.liveness_detector.is_live(frame)
        emotion = self.emotion_detector.predict(frame)
        identity, confidence = 'Unknown', 0.0
        
        embedding = self.extract_face_embedding(frame)
        identity, confidence = self.face_db.find_match(embedding)
        
        color = (255, 255, 255)
        conf_text = f"({confidence:.2f})" if identity != 'Unknown' else "(-)"
        cv2.putText(annotated_frame, f"Live: {is_live} ({liveness_score:.3f})", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)
        cv2.putText(annotated_frame, f"Emotion: {emotion.title()}", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)
        cv2.putText(annotated_frame, f"Identity: {identity} {conf_text}", (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)
        return annotated_frame
    

if __name__ == '__main__':
    liveness_detector = LivenessDetector()
    emotion_detector = EmotionDetector()
    img = cv2.imread('dataset/verification_data/00000001.jpg')
    print('Live:', liveness_detector.is_live(img))
    print('Emotion:', emotion_detector.predict(img))