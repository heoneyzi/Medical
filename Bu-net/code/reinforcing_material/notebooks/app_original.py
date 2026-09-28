import streamlit as st
from PIL import Image
import os

# 이미지 로드 함수
def load_images(image_dir, base_filename, count):
    images = {}
    for i in range(1, count + 1):
        filename = f"{base_filename}_{i}.png"
        image_path = os.path.join(image_dir, filename)
        if os.path.exists(image_path):
            images[i] = Image.open(image_path)
        else:
            st.error(f"File not found: {image_path}")
            return None
    return images

def main():
    image_dir = "test_data"  # 이미지 디렉토리 설정
    ground_truths = load_images(image_dir, "ground_truth", 50)
    model_predictions = load_images(image_dir, "model_prediction", 50)
    model2_predictions = load_images(image_dir, "model2_prediction", 50)
    model3_predictions = load_images(image_dir, "model3_prediction", 50)

    if ground_truths is not None:
        # 5x10 그리드로 이미지 표시
        idx = 1
        for i in range(5):
            cols = st.columns(10)
            for col in cols:
                if idx <= 50:
                    # 이미지 클릭 이벤트 처리
                    clicked = col.button("Image " + str(idx), key=idx)
                    if clicked:
                        st.session_state['selected_image'] = idx
                    col.image(ground_truths[idx], width=100, caption=f"Image {idx}")
                    idx += 1
        
        # 선택된 이미지에 대한 상세 정보 표시
        if 'selected_image' in st.session_state:
            selected_index = st.session_state['selected_image']
            st.write(f"Selected Image: {selected_index}")
            st.image(ground_truths[selected_index], caption="Ground Truth")
            st.image(model_predictions[selected_index], caption="Model 1 Prediction")
            st.image(model2_predictions[selected_index], caption="Model 2 Prediction")
            st.image(model3_predictions[selected_index], caption="Model 3 Prediction")

if __name__ == "__main__":
    main()