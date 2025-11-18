from PIL import Image
import image_augs as ia

ia.set_seed(42)

img = Image.open("example.jpg").convert("RGB")

# Простая аугментация
aug = Compose([
    ia.random_horizontal_flip(p=0.5),
    ia.random_color_jitter(0.2, 0.2, 0.2, 0.05, p=0.8),
    ia.random_gaussian_blur(max_radius=1.0, p=0.3),
])
img_aug = aug(img)

# Сильный дефолтный pipeline
strong_aug = ia.build_default_strong_aug(size=(224, 224))
img_aug2 = strong_aug(img)
