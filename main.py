import streamlit as st
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from detect import FaceRecognitionSystem
	
			
def run_streamlit_gui(embed_model_path):
	st.set_page_config(page_title='Enhanced Facial Recognition', layout='wide')
	st.title('Facial Recognition with Emotion & Liveness Detection')
	
	if 'system' not in st.session_state:
		st.session_state.system = FaceRecognitionSystem(embed_model_path)
	
	sys = st.session_state.system
	col1, col2 = st.columns([2, 1])
	
	with col1:
		class VideoProcessor(VideoProcessorBase):
			def recv(self, frame):
				img = frame.to_ndarray(format="bgr24")
				processed_img = sys.process_frame(img)
				new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
				return new_frame
			
		st.subheader('Live Detection')
		webrtc_streamer(
			key='face-recognition', video_processor_factory=VideoProcessor,
			rtc_configuration={'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]}
		)
	
	with col2:
		st.subheader('System Status')
		stats = sys.face_db.get_stats()
		liveness_threshold = sys.liveness_detector.threshold
		st.metric('Registered People', stats['total_people'])
		st.metric('Confidence Threshold', f"{stats['threshold']:.2f}")
		st.metric('Liveness Threshold', f'{liveness_threshold:.2f}')
		
		st.subheader('Instructions')
		st.info("""
		To register faces:
		1. Place .jpg images in the 'registered' folder
		2. Name files with person's name (e.g., 'john_doe.jpg')
		3. Restart the application to load new faces
		""")


if __name__ == '__main__':
	embedding_model = 'models/metric_embedding.keras'
	run_streamlit_gui(embedding_model)