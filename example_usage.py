# Reads the sa folder wiht dicom files and contours
# then draws the contours on the images.

from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from con2img import draw_contourmtcs2image as draw


image_folder = 'D:/BME/6felev/Onlab/sample/17660636AMR708/sa/images'
con_file = 'D:/BME/6felev/Onlab/sample/17660636AMR708/sa/contours.con'

# reading the dicom files
dr = DCMreaderVM(image_folder)

# reading the contours
cr = CONreaderVM(con_file)
contours = cr.get_hierarchical_contours()

# drawing the contours for the images
for slc in contours:
    for frm in contours[slc]:
        image = dr.get_image(slc, frm)  # numpy array
        cntrs = []
        rgbs = []
        for mode in contours[slc][frm]:
            # choose color
            if mode == 'ln':
                rgb = [1, 0, 0]
            elif mode == 'lp':
                rgb = [0, 1, 0]
            elif mode == 'rn':
                rgb = [1, 1, 0]
            else:
                rgb = None
            if rgb is not None:
                cntrs.append(contours[slc][frm][mode])
                rgbs.append(rgb)
        if len(cntrs) > 0:
            draw(image, cntrs, rgbs)
