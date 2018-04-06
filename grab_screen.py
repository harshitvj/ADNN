# Done by Frannecklp, modified a little.

import cv2
from roi import roi
import numpy as np
import win32gui
import win32ui
import win32con
import win32api


def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signed_ints_array = bmp.GetBitmapBits(True)
    img = np.fromstring(signed_ints_array, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def process_image():
    # grabbed_image = np.array(ImageGrab.grab(bbox=(0, 250, 960, 480)))
    grabbed_image = grab_screen(region=(0, 250, 960, 480))
    gray_image = cv2.cvtColor(grabbed_image, cv2.COLOR_BGR2GRAY)
    cropped_image = np.array(roi(gray_image))
    # screen = process_image(grabbed_image)
    # lines = cv2.HoughLinesP(screen, 1, np.pi/180, 100, None, minLineLength=150, maxLineGap=0)
    # screen = show_lines(cv2.cvtColor(grabbed_image,cv2.COLOR_BGR2RGB), lines)
    return np.array(cv2.resize(cropped_image, (96, 23)))
