"""camera_pid controller."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
import os
from keras.models import load_model
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# === Cargar modelo ===
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'modelo_nvidia_07.h5')
print("Modelo en:", model_path)

if os.path.exists(model_path):
    print("✅ Modelo cargado.")
    model = load_model(model_path, compile=False)
else:
    raise FileNotFoundError("❌ El modelo no existe.")

# === Funciones auxiliares ===

#Display image 
def display_image(display, image):
    # Image to display
    # image_rgb = np.dstack((image, image,image,))

    # Si la imagen es de un solo canal (escala de grises)
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Unsupported image format for display.")
    
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]  # quitar canal alfa

def preprocess_image(img):

    # processing display
    display_img_1 = Display("display_image_1")

    img = (img * 255).astype(np.uint8)
    h = img.shape[0]
    img[:h // 2, :, :] = 0  # blackout en la imagen completa

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    display_image(display_img_1, img)
    img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)

# === Lógica de movimiento ===

manual_steering = 0
steering_angle = 0.0
angle = 0.0
speed = 5.0  # velocidad inicial reducida
alpha = 0.2  # para suavizado

def set_steering_angle(wheel_angle):
    global angle, steering_angle
    steering_angle = wheel_angle

    # limitar valores absolutos
    if steering_angle > 0.5:
        steering_angle = 0.5
    elif steering_angle < -0.5:
        steering_angle = -0.5

    angle = steering_angle

def main():
    global steering_angle

    robot = Car()
    driver = Driver()

    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera_center")
    camera.enable(timestep)

    keyboard = Keyboard()
    keyboard.enable(timestep)

    last_photo_time = -float('inf')
    predicted_angle = 0.0

    while robot.step() != -1:
        image = get_image(camera)
        current_time = robot.getTime()

        # Controles manuales
        key = keyboard.getKey()
        if key == keyboard.UP:
            print("⬆️ Speed up")
        elif key == keyboard.DOWN:
            print("⬇️ Slow down")
        elif key == keyboard.RIGHT:
            print("➡️ Manual right")
        elif key == keyboard.LEFT:
            print("⬅️ Manual left")

        PHOTO_INTERVAL = 0.05 # predicciones más frecuentes

        if current_time - last_photo_time >= PHOTO_INTERVAL:
            last_photo_time = current_time

            processed_img = preprocess_image(image)
            predicted_angle_raw = model.predict(processed_img, verbose=0)[0][0]

            # Suavizado del ángulo
            predicted_angle = alpha * predicted_angle_raw + (1 - alpha) * predicted_angle

            print(f"Predicción bruta: {predicted_angle_raw:.4f}, Suavizada: {predicted_angle:.4f}")
            driver.setSteeringAngle(predicted_angle)

        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()
