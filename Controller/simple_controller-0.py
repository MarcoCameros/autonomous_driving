"""camera_pid controller for 66x200 images."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os

# Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Image processing
def greyscale_cv2(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur_cv2(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def canny_cv2(image):
    return cv2.Canny(image, threshold1=10, threshold2=20)

def roi_cv2(image):
    print(image.shape)

    ver = np.array([[(0,66),(0,60),(25,55),(50,50),(79,40),(160,40),(200,66)]], dtype=np.int32)
    

    roi_img = np.zeros_like(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        cv2.fillPoly(roi_img, ver, (255, 255, 255))
    else:
        cv2.fillPoly(roi_img, ver, 255)

    return cv2.bitwise_and(image, roi_img)

def average_line(lines):
    x1s, y1s, x2s, y2s = zip(*lines)
    return (
        int(np.mean(x1s)), int(np.mean(y1s)),
        int(np.mean(x2s)), int(np.mean(y2s))
    )

def hough_cv2(image, img_mask):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rho, theta, threshold = 3, np.pi/180, 10
    min_line_len, max_line_gap = 20, 30
    lines = cv2.HoughLinesP(img_mask, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    img_lines = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)

    left_lines, right_lines = [], []
    center_x = image.shape[1] // 2

    if lines is not None and len(lines) < 30:
        print(f"len lines: {len(lines)}")
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.2:
                    continue
                if x1 < center_x and x2 < center_x:
                    left_lines.append((x1, y1, x2, y2))
                    cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 4)
                elif x1 > center_x and x2 > center_x:
                    right_lines.append((x1, y1, x2, y2))
                    cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 4)

        if left_lines and right_lines:
            avg_left = average_line(left_lines)
            avg_right = average_line(right_lines)
            mid_x1 = int((avg_left[0] + avg_right[0]) / 2)
            mid_y1 = int((avg_left[1] + avg_right[1]) / 2)
            mid_x2 = int((avg_left[2] + avg_right[2]) / 2)
            mid_y2 = int((avg_left[3] + avg_right[3]) / 2)

            cv2.line(img_lines, (mid_x1, mid_y1), (mid_x2, mid_y2), (0, 0, 255), 6)

            error = center_x - (mid_x1 + 10)
            auto_steering = 0.005 * error
            set_steering_angle(-auto_steering)
            print(f"[INFO] error: {error}, steering: {auto_steering:.3f}")
    else:
        set_steering_angle(0)
        print("no line detected go straight")

    return cv2.addWeighted(img_rgb, 1, img_lines, 1, 1)

def display_image(display, image):
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Unsupported image format for display.")

    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 15

def set_speed(kmh):
    global speed

def set_steering_angle(wheel_angle):
    global angle, steering_angle
    delta = wheel_angle - steering_angle
    if delta > 0.1:
        wheel_angle = steering_angle + 0.1
    elif delta < -0.1:
        wheel_angle = steering_angle - 0.1

    steering_angle = max(min(wheel_angle, 0.5), -0.5)
    angle = steering_angle

def change_steer_angle(inc):
    global manual_steering
    new_steering = manual_steering + inc
    if -25.0 <= new_steering <= 25.0:
        manual_steering = new_steering
        set_steering_angle(manual_steering * 0.02)
    print("going straight" if manual_steering == 0 else f"turning {steering_angle:.2f} rad {'left' if steering_angle < 0 else 'right'}")

def main():
    manual_mode = False
    last_manual_input_time = 0
    MANUAL_TIMEOUT = 3.0

    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera_1 = robot.getDevice("camera_1")
    camera_1.enable(timestep)

    camera_2 = robot.getDevice("camera_1")
    camera_2.enable(timestep)

    display_img_1 = Display("display_image_1")
    display_img_2 = Display("display_image_2")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    last_photo_time = -float('inf')

    while robot.step() != -1:
        image = get_image(camera_1)
        grey = greyscale_cv2(image)
        blur = blur_cv2(grey)
        edges = canny_cv2(blur)
        roi = roi_cv2(edges)

        current_time = robot.getTime()
        key = keyboard.getKey()

        if key == keyboard.UP:
            set_speed(speed + 5.0)
        elif key == keyboard.DOWN:
            set_speed(speed - 5.0)
        elif key == keyboard.RIGHT:
            change_steer_angle(+1)
            manual_mode = True
            last_manual_input_time = current_time
        elif key == keyboard.LEFT:
            change_steer_angle(-1)
            manual_mode = True
            last_manual_input_time = current_time

        if manual_mode and (current_time - last_manual_input_time) > MANUAL_TIMEOUT:
            print("[INFO] Returning to auto mode")
            manual_mode = False

        hough_image = hough_cv2(image, roi) if not manual_mode else image

        display_image(display_img_1, roi)
        display_image(display_img_2, hough_image)

        if current_time - last_photo_time >= 1.0:
            last_photo_time = current_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            filename = f"{timestamp}.png"
            print("Image taken:", filename)
            image_path = os.path.join(os.getcwd(), filename)
            camera_2.saveImage(image_path, 1)
            with open("photos.csv", "a") as f:
                f.write(f"{image_path}, {angle}\n")

        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()