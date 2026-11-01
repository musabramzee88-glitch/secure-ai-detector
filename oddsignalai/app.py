import streamlit as st 
from ultralytics import YOLO 
import cv2 
import numpy as np 
from PIL import Image
import io

st.set_page_config(page_title="Secure AI & Privacy Shield", layout="centered")

st.title("🛡️ Secure AI Object Detection & Strict Privacy Shield")
st.write("تطبيق ذكاء اصطناعي مع درع حماية متطور: كشف الأجسام + تغبيش وجوه الأشخاص تلقائياً لحماية الخصوصية.")

def security_file_scanner(uploaded_file):
    try:
        if uploaded_file.size > 15 * 1024 * 1024:
            return False, "حجم الملف كبير جداً."
        bytes_data = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(bytes_data))
        img.verify()
        return True, "الملف آمن وصالح للمعالجة."
    except Exception:
        return False, "تحذير أمني: الملف غير صالح أو مشبوه."

file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if file:
    is_safe, security_message = security_file_scanner(file)
    
    if not is_safe:
        st.error(f"🚨 {security_message}")
    else:
        st.success(f"🔒 {security_message}")
        
        image = Image.open(file).convert("RGB") 
        st.image(image, caption="Original Image", width=600)
        
        if st.button("Detect Objects & Blur Faces"):
            with st.spinner("جارٍ معالجة الصورة وتطبيق درع الخصوصية..."):
                model = YOLO("yolov8m.pt") 
                results = model(image, conf=0.25)

                output = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                img_height, img_width, _ = output.shape

                # أولاً: المرور على الأجسام المكتشفة
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    name = model.names[class_id]

                    # إذا كان الجسم المكتشف هو شخص (Person)، نقوم بتغبيش الجزء العلوي منه (منطقة الوجه والرأس) تلقائياً!
                    if name == "person":
                        # تقدير منطقة الرأس (الـ 25% العلوي من مربّع الشخص)
                        box_height = y2 - y1
                        face_y2 = y1 + int(box_height * 0.45)
                        
                        # تطبيق الضبابية على منطقة الوجه/الرأس للشخص المكتشف
                        face_roi = output[y1:face_y2, x1:x2]
                        if face_roi.size > 0:
                            face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                            output[y1:face_y2, x1:x2] = face_roi
                            
                            cv2.putText(output, "Protected Privacy", (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # رسم باقي الأجسام العادية
                    label = f"{name} {confidence:.0%}"
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output, label, (x1, max(y1 - 25, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # ثانياً: محاولة استخدام الـ Haar Cascade كطبقة إضافية للوجوه الواضحة المقربة
                try:
                    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                    if not face_detector.empty():
                        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                        faces = face_detector.detectMultiScale(gray, 1.1, 4)
                        for (x, y, w, h) in faces:
                            face_roi = output[y:y+h, x:x+w]
                            face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                            output[y:y+h, x:x+w] = face_roi
                except Exception:
                    pass

                # عرض النتيجة النهائية
                st.image(
                    cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                    caption="Result: Objects Detected + Automatic Person Face Privacy Shield",
                    width=600
                )