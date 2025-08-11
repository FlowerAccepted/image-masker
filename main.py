import cv2
import numpy as np
import gradio as gr

def add_random_noise(img, noise_level=30):
    # img: numpy array, BGR
    noise = np.random.randint(-noise_level, noise_level + 1, img.shape, dtype='int16')
    noisy_img = img.astype('int16') + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')
    return noisy_img

def blur_edges(img, kernel_size=15):
    # 创建掩码，仅模糊边缘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    mask = cv2.dilate(edges, None, iterations=3)
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    mask = mask / 255.0

    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    mask = mask[..., np.newaxis]
    result = img * (1 - mask) + blurred * mask
    return result.astype('uint8')

def adjust_contrast(img, contrast):
    # contrast: 0.5~2.0，1.0为原始对比度
    img = img.astype(np.float32)
    img = (img - 127.5) * contrast + 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def process_image(input_img, noise_level, kernel_size, contrast):
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    img = adjust_contrast(img, contrast)
    noisy_img = add_random_noise(img, noise_level)
    result_img = blur_edges(noisy_img, kernel_size)
    return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="上传图片"),
        gr.Slider(0, 255, value=60, label="噪声强度"),
        gr.Slider(3, 51, value=15, step=2, label="模糊卷积核大小"),
        gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="对比度调整")
    ],
    outputs=gr.Image(type="numpy", label="处理后图片"),
    title="图片随机噪声与边缘模糊工具",
    description="上传图片，插入随机噪声、模糊边缘并可调整对比度。"
)

if __name__ == "__main__":
    demo.launch()