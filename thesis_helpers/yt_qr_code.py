

url = "https://www.youtube.com/watch?v=tKakeyL_8Fg"
import os
import qrcode
img = qrcode.make(url)
qr_code_filename = os.path.join("/Users/lucasdriessens/Documents/researchproject/thesis_helpers/qr_codes", "qr_code_yt_vid.png")
img.save(qr_code_filename)