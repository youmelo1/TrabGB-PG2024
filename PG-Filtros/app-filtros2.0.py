import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np

def overlay(background, foreground, x_offset=None, y_offset=None):
    test = background.shape
    
    if len(test) == 2:
        bg_h, bg_w = background.shape
        bg_channels = 3
    else:
        bg_h, bg_w, bg_channels = background.shape
        
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    if len(test) == 2:
        background_subsection = cv.cvtColor(background_subsection, cv.COLOR_GRAY2BGR)

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    if len(test) == 2:
        composite = cv.cvtColor(composite.astype(np.uint8), cv.COLOR_BGR2GRAY)
    
    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    out = background.copy()
    return out

def apply_rgb(img):
    return img

def apply_HSV(img):
    img_cv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return img_cv

def apply_canny(img):
    img_cv = cv.Canny(img, 100, 200)
    return img_cv

def apply_blur(img):
    img_cv = cv.blur(img, (20, 20))
    return img_cv

def apply_Gblur(img):
    img_cv = cv.GaussianBlur(img, (5, 5), 0)
    return img_cv

def apply_grayscale(img):
    img_cv = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img_cv

def apply_sharp(img):
    sharpen_filter = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    img_cv = cv.filter2D(img, ddepth=-1, kernel=sharpen_filter)
    return img_cv

def apply_inverted(img):
    img_cv = cv.bitwise_not(img)
    return img_cv

def apply_red(img):
    img_cv = img.copy()
    img_cv[:, :, (1, 2)] = 0
    return img_cv

def apply_green(img):
    img_cv = img.copy()
    img_cv[:, :, (0, 2)] = 0
    return img_cv

def apply_blue(img):
    img_cv = img.copy()
    img_cv[:, :, (0, 1)] = 0
    return img_cv

def apply_blackWhite(img):
    img_cv = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img_cv = cv.threshold(img_cv, 70, 255, cv.THRESH_BINARY)
    return img_cv

def apply_sticker(event, x, y, flags, param):
    global foreground, processed_img
    
    if event == cv.EVENT_LBUTTONDOWN:
        if param == 'boom':
            foreground = cv.imread('images/boom2.png', cv.IMREAD_UNCHANGED)
            foreground = cv.resize(foreground, (120, 120))
            processed_img = overlay(processed_img, foreground, x-40, y-40)
            
        elif param == 'yeah':
            foreground = cv.imread('images/yeah.png', cv.IMREAD_UNCHANGED)
            foreground = cv.resize(foreground, (120, 120))
            processed_img = overlay(processed_img, foreground, x-40, y-40)
            
        elif param == 'oculos':
            foreground = cv.imread('images/eyeglasses.png', cv.IMREAD_UNCHANGED)
            foreground = cv.resize(foreground, (120, 40))
            processed_img = overlay(processed_img, foreground, x-60, y-10)
            
        elif param == 'aranha':
            foreground = cv.imread('images/spider.png', cv.IMREAD_UNCHANGED)
            foreground = cv.resize(foreground, (120, 120))
            processed_img = overlay(processed_img, foreground, x-60, y-40)
            
        elif param == 'chapeu':
            foreground = cv.imread('images/cowboy_hat.png', cv.IMREAD_UNCHANGED)
            foreground = cv.resize(foreground, (180, 120))
            processed_img = overlay(processed_img, foreground, x-90, y-40)
            
        elif param == 'gato':
            foreground = cv.imread('images/cat.png', cv.IMREAD_UNCHANGED)
            foreground = cv.resize(foreground, (120, 120))
            processed_img = overlay(processed_img, foreground, x-60, y-40)
        
        elif param == 'rosto':
            foreground = cv.imread('images/smiley_face.png', cv.IMREAD_UNCHANGED)
            foreground = cv.resize(foreground, (120, 120))
            processed_img = overlay(processed_img, foreground, x-60, y-40)
        
        elif param == 'estrela':
            foreground = cv.imread('images/star.png', cv.IMREAD_UNCHANGED)
            foreground = cv.resize(foreground, (120, 120))
            processed_img = overlay(processed_img, foreground, x-60, y-40)
        
        cv.imshow('App', processed_img)

def save_image(img, count):
    filename = 'image' + str(count) + '.jpg'
    cv.imwrite(filename, img)
    return count + 1

def save_frame(img, count):
    filename = 'frame' + str(count) + '.jpg'
    cv.imwrite(filename, img)
    return count + 1

def using_camera():
    
    count = 0
    source = cv.VideoCapture(0)
    filter = apply_rgb
    while True:
        has_frame, frame = source.read()
        if not has_frame:
            break
        
        key = cv.waitKey(1)
        if key == 27:
            break       
        elif key == ord('1'):
            filter = apply_rgb
            filter = apply_HSV
        elif key == ord('2'):
            filter = apply_rgb
            filter = apply_canny
        elif key == ord('3'):
            filter = apply_rgb
            filter = apply_blur
        elif key == ord('4'):
            filter = apply_rgb
            filter = apply_Gblur
        elif key == ord('5'):
            filter = apply_rgb
            filter = apply_grayscale
        elif key == ord('6'):
            filter = apply_rgb
            filter = apply_sharp
        elif key == ord('7'):
            filter = apply_rgb
            filter = apply_inverted
        elif key == ord('8'):
            filter = apply_rgb
            filter = apply_red
        elif key == ord('9'):
            filter = apply_rgb
            filter = apply_green
        elif key == ord('0'):
            filter = apply_rgb
            filter = apply_blue
        elif key == ord('q'):
            filter = apply_rgb
            filter = apply_blackWhite
        elif key == ord('x'):
            filter = apply_rgb
        elif key == ord('s'):
            count = save_frame(filter(frame), count)
        
        cv.imshow('App', filter(frame))
        
    source.release()
    cv.destroyAllWindows()

def load_image():
    filepath = filedialog.askopenfilename()
    img = Image.open(filepath)
    img_cv = np.array(img) 
    return cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

def using_image():
    global processed_img
    img = load_image()
    img = cv.resize(img, (600, 600))
    processed_img = img.copy()
    count = 0
    while True:
        cv.imshow('App', processed_img)
        key = cv.waitKey(1)
        
        if key == ord('b'):
            cv.setMouseCallback('App', apply_sticker, param='boom')
        elif key == ord('y'):
            cv.setMouseCallback('App', apply_sticker, param='yeah')
        elif key == ord('o'):
            cv.setMouseCallback('App', apply_sticker, param='oculos')
        elif key == ord('a'):
            cv.setMouseCallback('App', apply_sticker, param='aranha')
        elif key == ord('g'):
            cv.setMouseCallback('App', apply_sticker, param='gato')
        elif key == ord('c'):
            cv.setMouseCallback('App', apply_sticker, param='chapeu')
        elif key == ord('r'):
            cv.setMouseCallback('App', apply_sticker, param='rosto')
        elif key == ord('e'):
            cv.setMouseCallback('App', apply_sticker, param='estrela')
        elif key == ord('1'):
            processed_img = apply_HSV(img)
        elif key == ord('2'):
            processed_img = apply_canny(img)
        elif key == ord('3'):
            processed_img = apply_blur(img)
        elif key == ord('4'):
            processed_img = apply_Gblur(img)
        elif key == ord('5'):
            processed_img = apply_grayscale(img)
        elif key == ord('6'):
            processed_img = apply_sharp(img)
        elif key == ord('7'):
            processed_img = apply_inverted(img)
        elif key == ord('8'):
            processed_img = apply_red(img)
        elif key == ord('9'):
            processed_img = apply_green(img)
        elif key == ord('0'):
            processed_img = apply_blue(img)
        elif key == ord('q'):
            processed_img = apply_blackWhite(img)
        elif key == ord('x'):
            processed_img = img.copy()
        elif key == ord('s'):
            count = save_image(processed_img, count)
        elif key == ord('n'):
            img = load_image()
            img = cv.resize(img, (600, 600))
            processed_img = img.copy()
        elif key == 27: 
            cv.destroyAllWindows()
            break

def handle_choice(choice):
    if choice == 1:
        using_camera()
    elif choice == 2:
        using_image()

def main():
    root = tk.Tk()
    root.geometry("250x100")

    button_camera = tk.Button(root, text="Usar Câmera", command=lambda: handle_choice(1))
    button_camera.pack(pady=5)

    button_image = tk.Button(root, text="Usar Imagem", command=lambda: handle_choice(2))
    button_image.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
