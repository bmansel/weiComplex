# Copyright 2021 Brad Mansel

######################################################################
# This file is part of weiComplex.
#
# weiComplex is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# weiComplex is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with weiComplex.  If not, see <https://www.gnu.org/licenses/>.
######################################################################

import time
import hdf5plugin
from numpy.core.fromnumeric import size
from pyFAI import azimuthalIntegrator
import PIL.Image as IImage
import pyFAI
import fabio
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import askdirectory, askopenfilename
from numpy.core.defchararray import multiply
#from ttkwidgets import CheckboxTreeview
from numpy.core.numeric import count_nonzero
from numpy.lib.twodim_base import triu_indices_from
from scipy import ndimage
import os
import subprocess
import sys
import matplotlib as plt
# plt.use('TkAgg')
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import tkinter.scrolledtext as tkscrolled
import numpy as np
from dataclasses import dataclass
#from pyFAI.units import R
from matplotlib import pyplot

window = Tk()

window.title("weiComplex")
width = window.winfo_screenwidth()
height = window.winfo_screenheight()
window.geometry("1200x700")
# window.iconbitmap('/home/brad/Dropbox/NSRRC_postdoc/code/weiComplex/weiComplex/nsrrc_app.ico')
#window.geometry("%dx%d" % (width, height))
#window.wm_attributes('-topmost', 1)
# window.withdraw()   # this supress the tk window
#window.attributes('-fullscreen', True)
# window.geometry('1600x750')


@dataclass(unsafe_hash=True)
class expData:
    """
    Scattering data class including metadata etc which is needed.
    q: numpy.array
    I: numpy.array
    err: numpy.array
    params is dictionary with all the other info needed
     """
    q: np.ndarray
    I: np.ndarray
    err: np.ndarray
    params: {}


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """  # this is amazing!!! not used
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(
        os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def call_pyFAI_calibration():
    window.update_idletasks()
    window.update()
    if varOS.get() == "Linux":
        os.system('pyFAI-calib2')
    elif varOS.get() == "Windows":
        program_dir = entAnacondaDir.get().strip()
        os.startfile(program_dir + "/" + 'pyFAI-calib2.exe')
        #messagebox.showinfo(title="Error", message="Calibration is not currently implemented in the binary version.\n Please install anaconda and in the anaconda cli: pip install pyFAI \n then run pyFAI-calib2.exe also in the anaconda cli")


def mask_image():
    file = filedialog.askopenfilename()
    if varOS.get() == "Linux":
        window.update_idletasks()
        os.system('pyFAI-drawmask ' + file)
    else:
        window.update_idletasks()
        window.update()
        os.startfile(entAnacondaDir.get().strip() +
                     "/" + 'pyFAI-drawmask.exe', file)
        #messagebox.showinfo(title="Error", message="Calibration is not currently implemented in the binary version.\n Please install anaconda and in the anaconda cli: pip install pyFAI \n then run pyFAI-drawmask.exe also in the anaconda cli")


def uniquify(path):
    counter = 0
    if os.path.exists(path):
        while True:
            counter += 1
            newName = path + str(counter)
            if os.path.exists(newName):
                continue
            else:
                path = newName
                break
    return path


def lastPath(path):
    counter = 0
    if os.path.exists(path):
        while True:
            counter += 1
            newName = path + str(counter)
            if os.path.exists(newName):
                continue
            else:
                if counter != 1:
                    path = path + str(counter-1)
                break
    return path


def openPONIfileWindow():
    global window
    # constants
    plankC = float(4.135667696e-15)  # Planck's constant in ev/Hz
    speedLightC = float(299792458)  # speed of light m/s

    # GUI first
    windowPONI = Toplevel(window)
    windowPONI.title("Make PONI file")
    windowPONI.geometry("1000x400")

    def readPoniFile():
        try:
            file = filedialog.askopenfilename(parent=windowPONI, initialdir=os.path(
                experimentDirectory), filetypes=[("PONI files", ".poni")])
        except:
            file = filedialog.askopenfilename(
                parent=windowPONI, filetypes=[("PONI files", ".poni")])
        if file is None:
            return
        ai = pyFAI.load(file)
        txtInfo.delete(1.0, END)
        txtInfo.insert(END, str(ai))

    def makePoniFile():
        global experimentDirectory

        paramsFIT2d = {"smpDetDist": entSmpDet.get().strip(), "beamCenX": entBeamCenX.get().strip(), "beamCenY": entBeamCenY.get().strip(),
                       "tilt": entTilt.get().strip(), "tiltPlanRot": entTiltPlanRot.get().strip(), "energy": entEnergy.get().strip()}

        for key in paramsFIT2d:
            if len(paramsFIT2d[key]) == 0:
                messagebox.showerror(
                    parent=windowPONI, title="Error", message=key + " not entered")
                return

        wavelength = plankC * speedLightC / 1000 / float(paramsFIT2d["energy"])
        #detector = pyFAI.decorators.Detector(varDetSel.get())
        ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(
            detector=varDetSel.get(), wavelength=wavelength)

        # Flip image for 1M!!!!!

        if varDetSel.get() == "eiger1m":
            paramsFIT2d["beamCenY"] = 1065.0 - float(paramsFIT2d["beamCenY"])

        ai.setFit2D(float(paramsFIT2d["smpDetDist"]), float(paramsFIT2d["beamCenX"]), float(paramsFIT2d["beamCenY"]),
                    tilt=float(paramsFIT2d["tilt"]), tiltPlanRotation=float(paramsFIT2d["tiltPlanRot"]))

        file = filedialog.asksaveasfile(
            parent=windowPONI, mode='w', defaultextension=".poni")
        if file is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        ai.write(file.name)

        # hack to get wavelength in file....
        #file = open(file.name,'r')
        #lines = file.readlines()
        # with open(file.name,'w') as f:
        #    for line in lines:
        #        f.write(line)
        #        f.write('\n')
        #    f.write("Wavelength: " + str(wavelength))
        # f.close()
        # end hack ...

    varDetSel = StringVar()
    varDetSel.set('eiger1m')
    # Entry box
    entSmpDet = Entry(windowPONI, width=10)
    entBeamCenX = Entry(windowPONI, width=10)
    entBeamCenY = Entry(windowPONI, width=10)
    entTilt = Entry(windowPONI, width=10)
    entTiltPlanRot = Entry(windowPONI, width=10)
    entEnergy = Entry(windowPONI, width=10)
    # option menu
    omDetSel = OptionMenu(windowPONI, varDetSel, 'eiger1m', 'eiger9m')
    # labels
    lblHeading = Label(
        windowPONI, text="Convert FIT2D parameters into PONI format and read .PONI using pyFAI geometry module", font=('TkDefaultFont', 15))
    lblsmpDet = Label(windowPONI, text="Direct Beam Dist. [mm]: ")
    lblBeamCenX = Label(windowPONI, text="Beam Center X [pix.]: ")
    lblBeamCenY = Label(windowPONI, text="Beam Center Y [pix.]: ")
    lblTilt = Label(windowPONI, text="Tilt [deg.]: ")
    lblTiltPlanRot = Label(windowPONI, text="Tilt Plan Rot. [deg.]: ")
    lblEnergy = Label(windowPONI, text="Energy [keV]: ")
    txtInfo = Text(windowPONI, width=120, height=5)

    btnMakePoni = Button(
        windowPONI, text="Save FIT2d params to PONI file", command=makePoniFile)
    btnReadPoni = Button(
        windowPONI, text="Read Poni File \n(show FIT2d params)", command=readPoniFile)

    lblHeading.grid(column=0, row=0, columnspan=30)
    omDetSel.grid(column=1, row=1, sticky='w')
    lblsmpDet.grid(column=0, row=2, sticky='e')
    entSmpDet.grid(column=1, row=2, sticky='w')
    lblBeamCenX.grid(column=0, row=3, sticky='e')
    entBeamCenX.grid(column=1, row=3, sticky='w')
    lblBeamCenY.grid(column=0, row=4, sticky='e')
    entBeamCenY.grid(column=1, row=4, sticky='w')
    lblTilt.grid(column=0, row=5, sticky='e')
    entTilt.grid(column=1, row=5, sticky='w')
    lblTiltPlanRot.grid(column=0, row=6, sticky='e')
    entTiltPlanRot.grid(column=1, row=6, sticky='w')
    lblEnergy.grid(column=0, row=7, sticky='e')
    entEnergy.grid(column=1, row=7, sticky='w')
    btnMakePoni.grid(column=0, row=8, columnspan=2)
    btnReadPoni.grid(column=4, row=8)

    txtInfo.grid(column=0, row=9, columnspan=30, rowspan=30, pady=10)


def readHeaderFile(fname):
    #global rigi, civi, expTime
    lines = []
    with open(os.path.join(experimentDirectory, fname + "002.txt")) as f:
        lines = f.readlines()

    numLines = 0
    for line in lines:
        numLines += 1
    numFrames = numLines / 7

    # get civi, rigi and exposure time values
    civi = []
    rigi = []
    expTime = []
    count = 0
    for line in lines:
        count += 1
        if count > numFrames and count <= 2*numFrames:
            civi.append(float(f'{line}'))
        if count > 2*numFrames and count <= 3*numFrames:
            rigi.append(float(f'{line}'))
        if count > 4*numFrames and count <= 5*numFrames:
            expTime.append(float(f'{line}'))
    f.close()
    return civi, rigi, expTime

# currently not used!
# #def centroidAirFrame():
#     #BeamCenX = 1440
#     #BeamCenY = 960

#     #find brightest pixel

#     BeamCenX = float(entBeamX.get().strip())
#     BeamCenY = float(entBeamY.get().strip())


#     pixels = 10 # half number of pixels used in centroid
#     airTransData = fabio.open(os.path.join(experimentDirectory,entTransAirFrm.get().strip() + "_master.h5")).data
#     #mask = fabio.open(os.path.join(experimentDirectory,txtMask1.get("1.0", 'end-1c'))).data
#     #mask = 1- mask # invert mask
#     #airTransData = np.multiply(mask,airTransData) # apply mask
#     newBeamCenter = ndimage.measurements.center_of_mass(airTransData[round(BeamCenY)-pixels:round(BeamCenY)+pixels,round(BeamCenX)-pixels:round(BeamCenX)+pixels])
#     newBeamCenter = [newBeamCenter[0] + round(BeamCenX)-pixels + 1, newBeamCenter[1] + round(BeamCenY)-pixels + 1] # need to correct for the size of the centroid image +1 as it starts at 0
#     print("New beam Center of air image: " + str(newBeamCenter))
#     entBeamX.delete(0,END) # clear the textbox
#     entBeamX.insert(END,str(round(newBeamCenter[0],8))+'\n')
#     entBeamY.delete(0,END) # clear the textbox
#     entBeamY.insert(END,str(round(newBeamCenter[1],8))+'\n')

def makeAIFit2d(**kwargs):
    energy = kwargs['energy']
    directBeam = kwargs['directBeam']
    beamX = kwargs['beamX']
    beamY = kwargs['beamY']
    tilt = kwargs['tilt']
    tiltPlanRotation = kwargs['tiltPlanRotation']
    detector = kwargs['detector']

    plankC = float(4.135667696e-15)  # Planck's constant in ev/Hz
    speedLightC = float(299_792_458)  # speed of light m/s

    wavelengthcalc = plankC * speedLightC / 1000 / float(energy)
    ai = azimuthalIntegrator.AzimuthalIntegrator(
        detector=detector, wavelength=wavelengthcalc)
    ai.setFit2D(float(directBeam), float(beamX), float(beamY),
                tilt=float(tilt), tiltPlanRotation=float(tiltPlanRotation))

    return ai


def makeFIT2dDic():
    FIT2dParams = {'directBeam': None, 'energy': None, 'beamX': None,
                   'beamY': None, 'tilt': None, 'tiltPlanRotation': None, 'detector': None}
    return FIT2dParams


def readSAXSpar(exp_dir, fname):
    FIT2dParams = makeFIT2dDic()
    FIT2dParams['detector'] = "eiger9m"
    FIT2dParams['tiltPlanRotation'] = 0.0
    FIT2dParams['tilt'] = 0.0
    lines = []
    count = 0
    with open(os.path.join(exp_dir, fname), encoding="utf8", errors='ignore') as f:
        lines = f.readlines()

    for count, line in enumerate(lines):
        if count == 3:
            FIT2dParams['energy'] = line.split()[0]
        if count == 8:
            FIT2dParams['beamX'] = line.split()[0]
            # 9m is in correct orientation both on same line
            FIT2dParams['beamY'] = line.split()[1]
        if count == 9:
            FIT2dParams['directBeam'] = line.split()[0]
    return FIT2dParams


def readWAXSpar(exp_dir, fname):
    FIT2dParams = makeFIT2dDic()
    FIT2dParams['detector'] = "eiger1m"
    lines = []
    count = 0
    with open(os.path.join(exp_dir, fname), encoding="utf8", errors='ignore') as f:
        lines = f.readlines()

    for count, line in enumerate(lines):
        if count == 0:
            FIT2dParams['directBeam'] = line.split()[0]
        if count == 1:
            FIT2dParams['energy'] = line.split()[0]
        if count == 2:
            FIT2dParams['beamX'] = line.split()[0]
        if count == 3:
            FIT2dParams['beamY'] = line.split()[0]
            # correct flip 2d image in theis definition
            FIT2dParams['beamY'] = 1065.0-float(FIT2dParams['beamY'])
        if count == 4:
            FIT2dParams['tiltPlanRotation'] = line.split()[0]
        if count == 5:
            FIT2dParams['tilt'] = line.split()[0]
    return FIT2dParams


def getAI(det_name):
    if det_name == "eiger9m":
        if varPsaxspar.get() == 1:
            Fit2dParams = readSAXSpar(experimentDirectory, "pSAXSpar.txt")
            ai = makeAIFit2d(**Fit2dParams)
        elif varPsaxspar.get() == 0:
            ai = pyFAI.load(experimentDirectory + "/" +
                            entPoni9M.get().strip())
    if det_name == "eiger1m":
        if varWaxspar.get() == 1:
            Fit2dParams = readWAXSpar(experimentDirectory, "WAXSpar.txt")
            ai = makeAIFit2d(**Fit2dParams)
        elif varWaxspar.get() == 0:
            ai = pyFAI.load(os.path.join(
                experimentDirectory, entPoni1M.get().strip()))
    return ai


def clickGetTransmission():
    getTransmission(entTransAirFrm.get().strip(),
                    entTransSmpFrm.get().strip(), 0)


def getTransmission(airTMImage, smpTMImage, isBKG):
    # get beam size
    # read in transmission images
    #mask = fabio.open(os.path.join(experimentDirectory,txtMask1.get("1.0", 'end-1c'))).data
    # mask = 1- mask # invert mask

    ai = getAI("eiger9m")  # always use 9m for transmission.
    airTransData = fabio.open(os.path.join(
        experimentDirectory, airTMImage + "_master.h5")).data
    # airTransData = np.multiply(airTransData,mask) #apply mask
    smpTransData = fabio.open(os.path.join(
        experimentDirectory, smpTMImage + "_master.h5")).data
    # smpTransData = np.multiply(smpTransData,mask) #apply mask
    Fit2dDic = ai.getFit2D()
    BeamCenX = Fit2dDic["centerX"]
    BeamCenY = Fit2dDic["centerY"]
    civiAir, rigiAir, expTimeAir = readHeaderFile(airTMImage)
    civiSmp, rigiSmp, expTimeSmp = readHeaderFile(smpTMImage)
    # check the center of beam
    pixels = 50  # half number of pixels used in centroid
    smpBeamCenter = ndimage.measurements.center_of_mass(smpTransData[round(
        BeamCenY)-pixels:round(BeamCenY)+pixels, round(BeamCenX)-pixels:round(BeamCenX)+pixels])
    # need to correct for the size of the centroid image +1 as it starts at 0
    smpBeamCenter = [smpBeamCenter[0] +
                     round(BeamCenX)-pixels + 1, smpBeamCenter[1] + round(BeamCenY)-pixels + 1]

    sumPix = [1, 2, 3]

    airInt = []
    smpIntSamCen = []  # using same beam center
    smpIntNewCen = []  # using same beam center
    smpTMSamCen = []
    smpTMNewCen = []
    for num in range(3):
        airInt.append(sum(sum(airTransData[round(BeamCenY)-sumPix[num]:round(
            BeamCenY)+sumPix[num], round(BeamCenX)-sumPix[num]:round(BeamCenX)+sumPix[num]]))/rigiAir[0])
        smpIntSamCen.append(sum(sum(smpTransData[round(BeamCenY)-sumPix[num]:round(
            BeamCenY)+sumPix[num], round(BeamCenX)-sumPix[num]:round(BeamCenX)+sumPix[num]]))/rigiSmp[0])
        smpIntNewCen.append(sum(sum(smpTransData[round(smpBeamCenter[1])-sumPix[num]:round(
            smpBeamCenter[1])+sumPix[num], round(smpBeamCenter[0])-sumPix[num]:round(smpBeamCenter[0])+sumPix[num]]))/rigiSmp[0])
        smpTMSamCen.append(round(smpIntSamCen[num]/airInt[num], 5))
        smpTMNewCen.append(round(smpIntNewCen[num]/airInt[num], 5))

    # options for the different beam sizes
    if isBKG == 0:
        if varBeamPixels.get() == "Beam 3 pixels":
            entsmpTMVal.delete(0, END)  # clear the textbox
            entsmpTMVal.insert(END, str(smpTMSamCen[0]))  # add the new TM
            smpTMFinal = smpTMSamCen[0]
        elif varBeamPixels.get() == "Beam 5 pixels":
            entsmpTMVal.delete(0, END)  # clear the textbox
            entsmpTMVal.insert(END, str(smpTMSamCen[1]))  # add the new TM
            smpTMFinal = smpTMSamCen[1]
        elif varBeamPixels.get() == "Beam 7 pixels":
            entsmpTMVal.delete(0, END)  # clear the textbox
            entsmpTMVal.insert(END, str(smpTMSamCen[2]))  # add the new TM
            smpTMFinal = smpTMSamCen[2]
    else:  # put in the background text box
        if varBeamPixels.get() == "Beam 3 pixels":
            entbkgTMVal.delete(0, END)  # clear the textbox
            entbkgTMVal.insert(END, str(smpTMSamCen[0]))  # add the new TM
            smpTMFinal = smpTMSamCen[0]
        elif varBeamPixels.get() == "Beam 5 pixels":
            entbkgTMVal.delete(0, END)  # clear the textbox
            entbkgTMVal.insert(END, str(smpTMSamCen[1]))  # add the new TM
            smpTMFinal = smpTMSamCen[1]
        elif varBeamPixels.get() == "Beam 7 pixels":
            entbkgTMVal.delete(0, END)  # clear the textbox
            entbkgTMVal.insert(END, str(smpTMSamCen[2]))  # add the new TM
            smpTMFinal = smpTMSamCen[2]

    # Output the different intensities to the command line
    print("Beamsizes with same beam center gave TM of: \n 3 pixel: " +
          str(smpTMSamCen[0]) + "\n 5 pixel: " + str(smpTMSamCen[1]) + "\n 7 pixel: " + str(smpTMSamCen[2]))
    print("Beamsizes with new beam center gave TM of: \n 3 pixel: " +
          str(smpTMNewCen[0]) + "\n 5 pixel: " + str(smpTMNewCen[1]) + "\n 7 pixel: " + str(smpTMNewCen[2]))

    #readHeaderFile(txtTransAir.get("1.0", 'end-1c'))
    return smpTMFinal
    # plt.figure()
    # plt.imshow(airTransData[round(BeamCenY)-20:round(BeamCenY)+20,round(BeamCenX)-20:round(BeamCenX)+20])
    # plt.show()

def import_reject_mask(exp_dir, reject_mask):
    data = np.loadtxt(exp_dir + "/" + reject_mask, usecols=(0,1))
    return data

def make_reject_mask(image, reject_data):
    # take x, y reject data and make 2d image mask
    #image[image == 1] = 0  # this is zero already
    image[ np.array(reject_data[:,1], dtype=np.int_) , np.array(reject_data[:,0], dtype=np.int_) ] = 1
    return np.array(image)

def apply_reject_mask(image, reject_data):
    #maskPixels = np.squeeze(np.where(image >= 4000000000))
    image[reject_data[:,1], np.array(reject_data[:,0]] = 0 # for image this is zero!
    return np.array(image)


def combine_masks(eiger_mask, user_mask, reject_mask):
    combined_mask = eiger_mask + user_mask + reject_mask
    combined_mask[combined_mask > 1] = 1
    #combined_mask[combined_mask > 1] = 1
    return np.array(combined_mask)

def make_all_masks(image, experimentDirectory, mask, detNum):
    if varEigerMask.get() == 1:
        eiger_mask = make_Eiger_mask(image)
    else: 
        eiger_mask = np.zeros(image.shape)
                
    if len(mask) > 0:
        user_mask = fabio.open(os.path.join(experimentDirectory, mask)).data
    else:
        user_mask = np.zeros(image.shape)

    if varReject.get() == 1 and detNum == 1: # detNum == 1 is SAXS!
        reject_data = import_reject_mask(experimentDirectory, "REJECT.dat")
        reject_mask = make_reject_mask(image, reject_data)
    else:
        reject_mask = np.zeros(image.shape)

    maskData = combine_masks(eiger_mask, user_mask, reject_mask)
    return maskData

def integrateImage(*args):  # fileImages, poni, mask,detNum, TM):
    # take in different number of arguments to handle background subtraction
    # Two cases without background subtraction (case 1) and with background subtraction (case2)
    # case 1: FileNameImage, det_name, mask, detNum, TM [varSubBkg =0]
    # case 2: FileNameSmp, FileNameBkg, det_name, mask, detNum, TM_smp, TM_bkg [varSubBkg =1]
    # data format [[0]name, [1]ID, [2]multi det. flag, [3]sum(I), [4]q, [5]I, [6]err]
    global scatteringData
    if scatteringData is None:
        imageCount = 0
        scatteringData = []
    else:
        imageCount = int(scatteringData[-1].params["ID"]) + 1

    if args[-1] == 0:
        fileNameSmp = args[0]
        det_name = args[1]
        mask = args[2]
        detNum = args[3]
        TMsmp = args[4]
        subBkg = args[5]
    else:
        fileNameSmp = args[0]
        fileNameBkg = args[1]
        det_name = args[2]
        mask = args[3]
        detNum = args[4]
        TMsmp = args[5]
        TMbkg = args[6]
        subBkg = args[7]

    # SAXS file name
    # fileNameImage
    # make directory to store files for SAXS
    if detNum == 1 or varDetector.get() == "1M":
        saveDir = uniquify(os.path.join(experimentDirectory,
                           fileNameSmp+"_WC"))  # only make for SAXS
        os.mkdir(saveDir)
    else:
        # Save WAXS in the same directory
        saveDir = lastPath(os.path.join(
            experimentDirectory, fileNameSmp+"_WC"))

    # read in the image
    if detNum == 1:
        imgSmp = fabio.open(os.path.join(
            experimentDirectory, fileNameSmp+"_master.h5"))
        if subBkg == 1:
            imgBkg = fabio.open(os.path.join(
                experimentDirectory, fileNameBkg+"_master.h5"))
    else:
        imgSmp = fabio.open(os.path.join(
            experimentDirectory, fileNameSmp[1:3]+fileNameSmp[0]+"_master.h5"))
        if subBkg == 1:
            imgBkg = fabio.open(os.path.join(
                experimentDirectory, fileNameSmp[1:3]+fileNameBkg[0]+"_master.h5"))
    numberFrames = imgSmp.nframes
    print("num frames: " + str(numberFrames))

    ai = getAI(det_name)
    # ai =pyFAI.load(os.path.join(experimentDirectory,poni)) # change to det_name

    # reset progress bar
    if detNum == 1 or varDetector.get() == "1M":
        progBarSAXS['value'] = 0
        progBarWAXS['value'] = 0

    civiSmp, rigiSmp, expTimeSmp = readHeaderFile(fileNameSmp)
    if subBkg == 1:
        civiBkg, rigiBkg, expTimeBkg = readHeaderFile(fileNameBkg)

    num_points = int(entNumPoints.get().strip())
    #intMethod = 'csr_ocl'
    intMethod = 'cython'
    pyFAI.use_opencl = False

    if varDetector.get() == "9M and 1M":
        multiDet = True
    else:
        multiDet = False

    for frame in range(numberFrames):

        # get background frame first if needed (only needed on first loop) and only 1 frame!
        window.update_idletasks()
        if subBkg == 1 and frame == 0:
            if detNum == 1:
                intImBkg = fabio.open(os.path.join(
                    experimentDirectory, fileNameBkg+"_master.h5"))
            else:
                intImBkg = fabio.open(os.path.join(
                    experimentDirectory, fileNameBkg[1:3]+fileNameBkg[0]+"_master.h5"))

        if numberFrames == 1:
            if detNum == 1:
                intImSmp = fabio.open(os.path.join(
                    experimentDirectory, fileNameSmp+"_master.h5"))
            else:
                intImSmp = fabio.open(os.path.join(
                    experimentDirectory, fileNameSmp[1:3]+fileNameSmp[0]+"_master.h5"))
        else:
            intImSmp = imgSmp.getframe(frame)
        if frame == 0: # first frame get the mask
            data_in = np.copy(intImSmp.data) # should be using copy as python objects are mutable!!!!
            maskData = make_all_masks(data_in, experimentDirectory, mask, detNum)
            print(intImSmp.data)

        # calc normalization value norm value is division
        # civi, thickness, TM, /scaleFactor
        normValueSmp = civiSmp[frame]*float(entThickness.get().strip()) * \
            TMsmp/float(entScaleFactor.get().strip())
        if subBkg == 1:
            normValueBkg = civiBkg[0]*float(entThickness.get().strip()) * \
                TMbkg/float(entScaleFactor.get().strip())

        if subBkg == 0:
            if detNum == 1:
                saveFileName = os.path.join(
                    saveDir, fileNameSmp+"_"+str(frame)+".dat")

            else:
                saveFileName = os.path.join(
                    saveDir, fileNameSmp+"W_"+str(frame)+".dat")
            if varSave.get() == "Save All Data":
                # after checking we can get standard error of mean by using the norm value
                q, I, err = ai.integrate1d(intImSmp.data, num_points, filename=saveFileName, correctSolidAngle=True, variance=None, error_model="poisson", radial_range=None, azimuth_range=None, mask=maskData,
                                           dummy=None, delta_dummy=None, polarization_factor=None, dark=None, flat=None, method=intMethod, unit='q_A^-1', safe=False, normalization_factor=normValueSmp, metadata=None)
            else:  # don't save
                q, I, err = ai.integrate1d(intImSmp.data, num_points, correctSolidAngle=True, variance=None, error_model="poisson", radial_range=None, azimuth_range=None, mask=maskData,
                                           dummy=None, delta_dummy=None, polarization_factor=None, dark=None, flat=None, method=intMethod, unit='q_A^-1', safe=False, normalization_factor=normValueSmp, metadata=None)

            if detNum == 1:
                is1M = False
                #scatteringData.append([fileNameSmp+"_"+str(frame), imageCount, multiDet, np.sum(I), q, I, err, TMsmp, is1M,fileNameSmp,frame])
                params = {"file name": fileNameSmp+"_"+str(frame), "ID": imageCount, "multiDet": multiDet, "sumI": np.sum(
                    I), "TM": TMsmp, "is1M": is1M, "fileNameSmp": fileNameSmp, "frame": frame}
                scatteringData.append(expData(q, I, err, params))
            else:
                is1M = True
                params = {"file name": fileNameSmp+"W_"+str(frame), "ID": imageCount, "multiDet": multiDet, "sumI": np.sum(
                    I), "TM": TMsmp, "is1M": is1M, "fileNameSmp": fileNameSmp, "frame": frame}
                scatteringData.append(expData(q, I, err, params))

                #scatteringData.append([fileNameSmp+"W_"+str(frame), imageCount, multiDet, np.sum(I), q, I, err, TMsmp, is1M,fileNameSmp[1:3]+fileNameSmp[0], frame])

        else:  # subtract background step
            if detNum == 1:
                is1M = False
                saveFileNameSmp = os.path.join(
                    saveDir, fileNameSmp+"_SUB_"+str(frame)+".dat")
                #saveFileNameBkg = os.path.join(saveDir,"BKG_"+fileNameBkg+"_"+str(frame)+".dat")
                q, ISmp, errSmp = ai.integrate1d(intImSmp.data, num_points, correctSolidAngle=True, variance=None, error_model="poisson", radial_range=None, azimuth_range=None, mask=maskData,
                                                 dummy=None, delta_dummy=None, polarization_factor=None, dark=None, flat=None, method=intMethod, unit='q_A^-1', safe=False, normalization_factor=normValueSmp, metadata=None)
                if frame == 0:  # get the first frame
                    q, IBkg, errBkg = ai.integrate1d(intImBkg.data, num_points, correctSolidAngle=True, variance=None, error_model="poisson", radial_range=None, azimuth_range=None, mask=maskData,
                                                     dummy=None, delta_dummy=None, polarization_factor=None, dark=None, flat=None, method=intMethod, unit='q_A^-1', safe=False, normalization_factor=normValueBkg, metadata=None)
                # [0] file name (for save), [1] id_num, [2] multiDet_flag,[3] sum(1), [4] q, [5] I, [6] err, [7] TM, [8] 1M flag, [9] filename, [10] frame
                # error propagation for subtracting two numbers = sqroot(sigmaA^2 +sigmaB^2)
                err = np.sqrt(np.add(np.power(errSmp, 2), np.power(errBkg, 2)))
                I = np.subtract(ISmp, IBkg)
                params = {"file name": fileNameSmp+"SUB_"+str(frame), "ID": imageCount, "multiDet": multiDet, "sumI": np.sum(
                    I), "TM": TMsmp, "is1M": is1M, "fileNameSmp": fileNameSmp, "frame": frame}
                scatteringData.append(expData(q, I, err, params))
                #scatteringData.append([fileNameSmp+"SUB_"+str(frame), imageCount, multiDet, np.sum(I), q, I, err, TMsmp, is1M,fileNameSmp,frame])
                # write data to file
                if varSave.get() == "Save All Data":
                    np.savetxt(saveFileNameSmp, np.transpose(
                        [q, I, err]), fmt='%1.6e', delimiter='    ')

            else:
                is1M = True
                saveFileNameSmp = os.path.join(
                    saveDir, fileNameSmp+"_SUB_W_"+str(frame)+".dat")
                q, ISmp, errSmp = ai.integrate1d(intImSmp.data, num_points, correctSolidAngle=True, variance=None, error_model="poisson", radial_range=None, azimuth_range=None, mask=maskData,
                                                 dummy=None, delta_dummy=None, polarization_factor=None, dark=None, flat=None, method=intMethod, unit='q_A^-1', safe=False, normalization_factor=normValueSmp, metadata=None)
                if frame == 0:  # get the first frame
                    q, IBkg, errBkg = ai.integrate1d(intImBkg.data, num_points, correctSolidAngle=True, variance=None, error_model="poisson", radial_range=None, azimuth_range=None, mask=maskData,
                                                     dummy=None, delta_dummy=None, polarization_factor=None, dark=None, flat=None, method=intMethod, unit='q_A^-1', safe=False, normalization_factor=normValueBkg, metadata=None)
                err = np.sqrt(np.add(np.power(errSmp, 2), np.power(errBkg, 2)))
                I = np.subtract(ISmp, IBkg)
                #scatteringData.append([fileNameSmp+"SUB_W_"+str(frame), imageCount, multiDet, np.sum(I), q, I, err, TMsmp, is1M,fileNameSmp[1:3]+fileNameSmp[0], frame])
                params = {"file name": fileNameSmp+"SUB_W_"+str(frame), "ID": imageCount, "multiDet": multiDet, "sumI": np.sum(
                    I), "TM": TMsmp, "is1M": is1M, "fileNameSmp": fileNameSmp, "frame": frame}
                scatteringData.append(expData(q, I, err, params))

                # write data to file
                if varSave.get() == "Save All Data":
                    np.savetxt(saveFileNameSmp, np.transpose(
                        [q, I, err]), fmt='%1.6e', delimiter='    ')
        # make unique ID with A and B for the different detectors im multi detector mode

        tvData.insert(parent='', index='end', iid=scatteringData[-1].params["ID"], text="", values=(scatteringData[-1].params["ID"],
                      scatteringData[-1].params["file name"], round(scatteringData[-1].params["sumI"], 5), round(scatteringData[-1].params["TM"], 4)))
        imageCount += 1
        window.update_idletasks()
    # with open('experiment.wc', 'w') as outfile:
      ######################################
      # FIIIIIIIIIXXXXXXXXXXXXXXXXXXX

        if detNum == 1:
            progBarSAXS['value'] += 100*(1/numberFrames)
        elif detNum == 2:
            progBarWAXS['value'] += 100*(1/numberFrames)
        window.update_idletasks()
        window.update()
    # repopulate()
    try:
        np.array(scatteringData, dtype=object).dump(open(os.path.join(
            experimentDirectory, entExperimentFile.get().strip()), 'wb'))  # save the file
    except:
        messagebox.showinfo(
            title="Error", message="Save data location unknown, \nplease select file > save experiment \n or check advanced tab has the data file with .wc ")

def integrateMultiDet():
    global entTransAirFrm, entBkgSAXSFrm, entTransBkgFrm, entTransSmpFrm, entSmpSAXSFrm
    # case 1: FileNameImage, det_name, mask, detNum, TM, [varSubBkg =0]
    # case 2: FileNameSmp, FileNameBkg, det_name, mask, detNum, TM_smp, TM_bkg, [varSubBkg =1]

    for image in range(len(entSmpSAXSFrm)):
        print("frame number: " + str(len(entSmpSAXSFrm)))
        if varCalcTMSmp.get() == 1:  # if check box is slected calc TM will update GUI once function finishes
            TMsmp = getTransmission(entTransAirFrm[image].get().strip(
            ), entTransSmpFrm[image].get().strip(), 0)  # calculate the sample transmisison
            # txtsmpTMVal.insert(str(round(TMsmp,5))) # gets put in by other function
            window.update_idletasks()
        else:
            window.update_idletasks()
            # otherwise get smp TM from GUI
            TMsmp = float(entsmpTMVal.get().strip())

        # do same for background!
        # if check box is slected calc TM will update GUI once function finishes
        if varCalcTMbkg.get() == 1 and varSubBkg[image].get() == 1:
            TMbkg = getTransmission(entTransAirFrm[image].get().strip(
            ), entTransBkgFrm[image].get().strip(), 1)  # calculate the background transmisison
            # txtbkgTMVal.insert(str(round(TMbkg,5))) # gets put in by the other function!
            window.update_idletasks()
        elif varCalcTMbkg.get() == 0 and varSubBkg[image].get() == 1:
            window.update_idletasks()
            # otherwise get bkg TM from GUI
            TMbkg = float(entbkgTMVal.get().strip())

        # do the actual integration(s)
        if varSubBkg[image].get() == 0:
            if varDetector.get() == "9M":  # no subtract background 9M only
                integrateImage(entSmpSAXSFrm[image].get().strip(
                ), "eiger9m", entMask9M.get().strip(), 1, TMsmp, varSubBkg[image].get())
            elif varDetector.get() == "1M":  # no subtract background 1M only
                integrateImage(entSmpSAXSFrm[image].get().strip(
                ), "eiger1m", entMask1M.get().strip(), 2, TMsmp, varSubBkg[image].get())
            elif varDetector.get() == "9M and 1M":  # no subtract background 9M and 1M
                integrateImage(entSmpSAXSFrm[image].get().strip(
                ), "eiger9m", entMask9M.get().strip(), 1, TMsmp, varSubBkg[image].get())
                integrateImage(entSmpSAXSFrm[image].get().strip(
                ), "eiger1m", entMask1M.get().strip(), 2, TMsmp, varSubBkg[image].get())

        # subtract background, FileNameSmp, FileNameBkg, poni, mask, detNum, TM_smp, TM_bkg
        elif varSubBkg[image].get() == 1:
            if varDetector.get() == "9M":  # sub bkg 9M only
                integrateImage(entSmpSAXSFrm[image].get().strip(), entBkgSAXSFrm[image].get().strip(
                ), "eiger9m", entMask9M.get().strip(), 1, TMsmp, TMbkg, varSubBkg[image].get())
            elif varDetector.get() == "1M":  # sub bkg 1M only
                integrateImage(entSmpSAXSFrm[image].get().strip(), entBkgSAXSFrm[image].get().strip(
                ), "eiger1m", entMask1M.get().strip(), 2, TMsmp, TMbkg, varSubBkg[image].get())
            elif varDetector.get() == "9M and 1M":  # sub bkg 9M and 1M
                integrateImage(entSmpSAXSFrm[image].get().strip(), entBkgSAXSFrm[image].get().strip(
                ), "eiger9m", entMask9M.get().strip(), 1, TMsmp, TMbkg, varSubBkg[image].get())
                integrateImage(entSmpSAXSFrm[image].get().strip(), entBkgSAXSFrm[image].get().strip(
                ), "eiger1m", entMask1M.get().strip(), 2, TMsmp, TMbkg, varSubBkg[image].get())


def saveParams():
    global experimentDirectory
    lines = ["TM Air FRM, " + entTransAirFrm[0].get(),
             "TM smp FRM, " + entTransSmpFrm[0].get().strip(),
             "TM bkg FRM, " + entTransBkgFrm[0].get().strip(),
             "TM smp VAL, " + entsmpTMVal.get().strip(),
             "TM bkg VAL, " + entbkgTMVal.get().strip(),
             "smpSAXS FRM, " + entSmpSAXSFrm[0].get().strip(),
             "bkgSAXS FRM, " + entBkgSAXSFrm[0].get().strip(),
             "Scale factor, " + entScaleFactor.get().strip(),
             "Mask 9M, " + entMask9M.get().strip(),
             "Mask 1M, " + entMask1M.get().strip(),
             "PONI 9M, " + entPoni9M.get().strip(),
             "PONI 1M, " + entPoni1M.get().strip(),
             "Thickness, " + entThickness.get().strip(),
             "Num. points, " + entNumPoints.get().strip(),
             # advanced settings
             "OS, " + varOS.get(),
             "primusDir, " + entPrimusDir.get().strip(),
             "experimentFile, " + entExperimentFile.get().strip(),
             "experimentDirectory, " + str(experimentDirectory)
             ]
    #"anacondaDir, "  + txtAnacondaDir.get("1.0", 'end-1c')
    with open(os.path.join(experimentDirectory, 'params.wc'), 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    f.close()

# use , and split to get the first part and second part then put the save values in the correct places


def loadParams():
    global experimentDirectory
    global scatteringData
    fileName = askopenfilename(filetypes=[("WeiComplex param", ".wc")])
    experimentDirectory = os.path.dirname(fileName)
    lblDirectory.config(text="Directory: "+experimentDirectory)
    file = open(fileName, 'r')
    lines = file.readlines()
    for line in lines:
        if line.split(",")[0] == "TM Air FRM":
            entTransAirFrm[0].delete(0, END)
            entTransAirFrm[0].insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "TM smp FRM":
            entTransSmpFrm[0].delete(0, END)
            entTransSmpFrm[0].insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "TM bkg FRM":
            entTransBkgFrm[0].delete(0, END)
            entTransBkgFrm[0].insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "TM smp VAL":
            entsmpTMVal.delete(0, END)
            entsmpTMVal.insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "smpSAXS FRM":
            entSmpSAXSFrm[0].delete(0, END)
            entSmpSAXSFrm[0].insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "bkgSAXS FRM":
            entBkgSAXSFrm[0].delete(0, END)
            entBkgSAXSFrm[0].insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "Scale factor":
            entScaleFactor.delete(0, END)
            entScaleFactor.insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "Mask 9M":
            entMask9M.delete(0, END)
            entMask9M.insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "Mask 1M":
            entMask1M.delete(0, END)
            entMask1M.insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "PONI 9M":
            entPoni9M.delete(0, END)
            entPoni9M.insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "PONI 1M":
            entPoni1M.delete(0, END)
            entPoni1M.insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "Thickness":
            entThickness.delete(0, END)
            entThickness.insert(END, line.split(",")[1].strip())

        # if line.split(",")[0] == "Beam Cen. X":
        #     entBeamX.delete(0, END)
        #     entBeamX.insert(END,line.split(",")[1].strip())

        # if line.split(",")[0] == "Beam Cen. Y":
        #     entBeamY.delete(0, END)
        #     entBeamY.insert(END,line.split(",")[1].strip())

        if line.split(",")[0] == "Num. points":
            entNumPoints.delete(0, END)
            entNumPoints.insert(END, line.split(",")[1].strip())
            ent9MEnd.delete(0, END)
            ent9MEnd.insert(END, line.split(",")[1].strip())
            ent1MEnd.delete(0, END)
            ent1MEnd.insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "OS":
            varOS.set(line.split(",")[1].strip())

        if line.split(",")[0] == "primusDir":
            entPrimusDir.delete(0, END)
            entPrimusDir.insert(END, line.split(",")[1].strip())

        if line.split(",")[0] == "experimentFile":
            entExperimentFile.delete(0, END)
            entExperimentFile.insert(END, line.split(",")[1].strip())
            try:
                scatteringDataTemp = np.load(open(os.path.join(
                    experimentDirectory, entExperimentFile.get().strip()), 'rb'), allow_pickle=True)
                for item in scatteringDataTemp:
                    if scatteringData is None:
                        scatteringData = []
                    scatteringData.append(item)
                    tvData.insert(parent='', index='end', iid=item.params["ID"], text="", values=(item.params["ID"], item.params["file name"],
                                                                                                  round(item.params["sumI"], 5), round(item.params["TM"], 4)))
                messagebox.showinfo(title="Data file read into memory", message=str(
                    len(scatteringData)) + " 1d profiles loaded")
            except:
                messagebox.showinfo(title="Data file not found",
                                    message="No file, 0 profiles loaded")


#np.load(open(os.path.join(experimentDirectory, 'experiment.npy'), 'rb'),allow_pickle=True)

def averageSelected(event=None):
    global scatteringData
    # be very careful we are getting indeces from the treeview and using with the total data matrix
    num_av = 0
    I_sum = []
    I_sum_err = []
    total_record = []
    for record in tvData.selection():
        x = tvData.item(record)["values"][0]
        if num_av == 0:
            multiDetFlag = scatteringData[int(x)].params["multiDet"]
            is1M = scatteringData[int(x)].params["is1M"]
            fileName = scatteringData[int(x)].params["file name"]
            frame = scatteringData[int(x)].params["frame"]
            q = scatteringData[int(x)].q
            I_sum = np.zeros(np.shape(scatteringData[int(x)].I))
            I_sum_err = np.zeros(np.shape(scatteringData[int(x)].err))
        total_record.append(int(x))
        I_sum = np.add(I_sum, scatteringData[int(x)].I)
        I_sum_err = np.add(I_sum_err, np.power(scatteringData[int(x)].err, 2))
        num_av += 1
    I_av = np.divide(I_sum, num_av)
    err_av = np.divide(np.sqrt(I_sum_err), num_av)  # error is divided here

    # displaying TM  doesn't make sense for averaged data so I will set it to 1.0
    params = {"file name": scatteringData[total_record[0]].params["file name"]+"_av_"+str(num_av)+"_frms", "ID": int(scatteringData[-1].params["ID"]) + 1,
              "multiDet": multiDetFlag, "sumI": np.sum(I_av), "TM": 1.0, "is1M": is1M, "fileNameSmp": fileName, "frame": frame}
    scatteringData.append(expData(q, I_av, err_av, params))
    # scatteringData.append([scatteringData[total_record[0]]["file name"]+"_av_"+str(num_av)+"_frms", int(scatteringData[-1]["ID"]) + 1, multiDetFlag, np.sum(I_av), q, I_av, err_av,1.0,is1M,fileName,frame]) # update scattering data
    # save the file
    try:
        np.array(scatteringData, dtype=object).dump(
            open(os.path.join(experimentDirectory, entExperimentFile.get()), 'wb'))
        tvData.insert(parent='', index='end', iid=scatteringData[-1].params["ID"], text="", values=(scatteringData[-1].params["ID"],
                                                                                                    scatteringData[-1].params["file name"], round(scatteringData[-1].params["sumI"], 5), round(scatteringData[-1].params["TM"], 4)))  # update treeview data
    except:
        messagebox.showinfo(
            title="Error", message="Save data location unknown, \nplease select file > save experiment \n or check advanced tab has the data file with .wc \nNo data processing performed.")


def subBackground(event=None):
    global backGroundID
    multiDetFlag = scatteringData[int(backGroundID)].params["multiDet"]
    is1M = scatteringData[int(backGroundID)].params["is1M"]
    q = scatteringData[int(backGroundID)].q
    I_bkg = np.multiply(
        scatteringData[int(backGroundID)].I, float(entScaleBkg.get()))
    err_bkg = np.multiply(
        scatteringData[int(backGroundID)].err, float(entScaleBkg.get()))
    for record in tvData.selection():
        I_smp = np.multiply(
            scatteringData[int(record)].I, float(entScaleSmp.get()))
        err_smp = np.multiply(
            scatteringData[int(record)].err, float(entScaleSmp.get()))
        I = np.subtract(I_smp, I_bkg)
        err = np.sqrt(np.add(np.power(err_smp, 2), np.power(err_bkg, 2)))
        fileName = scatteringData[int(record)].params["file name"]
        frame = scatteringData[int(record)].params["frame"]
        # displaying TM  doesn't make sense for subtracted data so I will set it to 1.0
        params = {"file name": scatteringData[int(record)].params["file name"]+"_SUB_"+str(record), "ID": scatteringData[-1].params["ID"]+1,
                  "multiDet": multiDetFlag, "sumI": np.sum(I), "TM": 1.0, "is1M": is1M, "fileNameSmp": fileName, "frame": frame}
        scatteringData.append(expData(q, I, err, params))
        # save the file
        try:
            np.array(scatteringData, dtype=object).dump(
                open(os.path.join(experimentDirectory, entExperimentFile.get().strip()), 'wb'))
            tvData.insert(parent='', index='end', iid=scatteringData[-1].params["ID"], text="", values=(scatteringData[-1].params["ID"],
                                                                                                        scatteringData[-1].params["file name"], round(scatteringData[-1].params["sumI"], 5), round(scatteringData[-1].params["TM"], 4)))  # update treeview data
        except:
            messagebox.showinfo(
                title="Error", message="Save data location unknown, \nplease select file > save experiment \n or check advanced tab has the data file with .wc\nNo data processing performed")


def selectBackground(event=None):
    global scatteringData
    global backGroundID
    x = tvData.item(tvData.selection())["values"][0]
    backGroundName = scatteringData[int(x)].params["file name"]
    backGroundID = scatteringData[int(x)].params["ID"]
    lblBackground.config(text="Background Frame: " +
                         backGroundName + " ID is " + str(backGroundID))


def plotSelectedData(event=None):
    global scatteringData, graph_past, fig, plot1, canvas
    plot1.clear()

    if len(tvData.selection()) > 20:
        ans = messagebox.askquestion(
            title="Continue?", message="It is best to plot batches of 10-20, for greater than 20 causes hanging, continue?")
        if ans == 'no':
            return

    # if Autoscale selected check only 1 1M and 1 9M
    if varAutoScale.get() == 1:
        if len(tvData.selection()) != 2:
            messagebox.showerror(
                'Error', 'Can only auto scale with 1 x 1M and 1x9M')
            return
        flag1M = []
        ID = []
        for record in tvData.selection():
            x = tvData.item(record)["values"][0]
            ID.append(x)
            flag1M.append(scatteringData[int(x)].params["is1M"])
        if (flag1M[0] == flag1M[1]):
            messagebox.showerror(
                'Error', 'Only 1 1M and 1 9M frame should be selected.')
            return
        # get 9M and 1M ID
        if flag1M[0]:
            ID1M = ID[0]
            ID9M = ID[1]
        else:
            ID1M = ID[1]
            ID9M = ID[0]

        start_index_9M = int(ent9MStart.get().strip())
        end_index_9M = int(ent9MEnd.get().strip())
        start_index_1M = int(ent1MStart.get().strip())
        end_index_1M = int(ent1MEnd.get().strip())

        overlap_1M_indices, overlap_9M_indices = find_overlap(
            scatteringData[ID9M].q[start_index_9M:end_index_9M], scatteringData[ID1M].q[start_index_1M:end_index_1M])
        scale1M = calc_scale_factor(
            scatteringData[ID9M], scatteringData[ID1M], overlap_1M_indices, overlap_9M_indices)
        entScale1M.delete(0, END)
        entScale1M.insert(END, str(round(scale1M, 8)))

    elif varAutoScale.get() == 0:
        scale1M = float(entScale1M.get().strip())

    for record in tvData.selection():
        x = tvData.item(record)["values"][0]
        # get iid to plot that data

        # first check if it is merged data
        if scatteringData[int(x)].params["is1M"] == False:
            try:
                if scatteringData[int(x)].params["isMerge"]:
                    first = 0
                    last = len(scatteringData[int(x)].q)
            except:
                first = int(ent9MStart.get())
                last = int(ent9MEnd.get())
            scale = 1.0
        elif scatteringData[int(x)].params["is1M"] == True:
            first = int(ent1MStart.get())
            last = int(ent1MEnd.get())
            scale = scale1M

        if varErrorBar.get() == 1:
            if varPlotStyle.get() == "Line":
                plot1.errorbar(scatteringData[int(x)].q[first:last], np.multiply(scatteringData[int(x)].I[first:last], scale),
                               yerr=np.multiply(
                                   scatteringData[int(x)].err[first:last], scale),
                               label=scatteringData[int(x)].params["file name"], linewidth=float(entLineWidth.get()), elinewidth=float(entLineWidth.get()))
            elif varPlotStyle.get() == "Points":
                plot1.errorbar(scatteringData[int(x)].q[first:last], np.multiply(scatteringData[int(x)].I[first:last], scale),
                               yerr=np.multiply(scatteringData[int(x)].err[first:last], scale), linestyle='none',
                               label=scatteringData[int(x)].params["file name"], markersize=float(entLineWidth.get()), elinewidth=float(entLineWidth.get()), marker='.')

        else:
            if varPlotStyle.get() == "Line":
                plot1.plot(scatteringData[int(x)].q[first:last], np.multiply(scatteringData[int(x)].I[first:last], scale), label=scatteringData[int(x)].params["file name"],
                           linewidth=float(entLineWidth.get()), elinewidth=float(entLineWidth.get()))
            elif varPlotStyle.get() == "Points":
                plot1.plot(scatteringData[int(x)].q[first:last], np.multiply(scatteringData[int(x)].I[first:last], scale),
                           linestyle='none', label=scatteringData[int(x)].params["file name"], markersize=float(entLineWidth.get()), marker='.')

    if varLogX.get() == 1:
        plot1.set_xscale('log')

    if varLogY.get() == 1:
        plot1.set_yscale('log')

    plot1.set_xlabel(r'q [$\AA$]')
    plot1.set_ylabel("I(q)")
    plot1.legend(fontsize=8)
    plot1.tick_params(labelsize=8)
    if varGrid.get() == 1:
        plot1.grid(which='both')
    elif varGrid.get() == 0:
        plot1.grid(False)
    fig.tight_layout()
    canvas.draw()
    lblstatus.config(text="Status: plotted selected data.")


def saveSelection(event=None):
    global scatteringData
    global experimentDirectory
    for record in tvData.selection():
        x = tvData.item(record)["values"][0]
        saveName = scatteringData[int(x)].params["file name"]
        q = scatteringData[int(x)].q
        I = scatteringData[int(x)].I
        err = scatteringData[int(x)].err
        if varExportas.get() == 1:
            file = filedialog.asksaveasfile(
                parent=window, mode='w', initialfile=saveName + ".dat", defaultextension=".dat")
            if file is None:  # asksaveasfile return `None` if dialog closed with "cancel".
                return
            np.savetxt(file.name, np.transpose(
                [q, I, err]), fmt='%1.6e', delimiter='    ')
        elif varExportas.get() == 0:
            np.savetxt(os.path.join(experimentDirectory, saveName + ".dat"),
                       np.transpose([q, I, err]), fmt='%1.6e', delimiter='    ')

    lblstatus.config(text="Status: exported selected frames.")


def primusOpen():
    window.update_idletasks()
    window.update()
    global scatteringData
    files = []
    for record in tvData.selection():
        x = tvData.item(record)["values"][0]
        # first save entries
        saveName = scatteringData[int(x)].params["file name"]
        q = scatteringData[int(x)].q
        I = scatteringData[int(x)].I
        err = scatteringData[int(x)].err
        np.savetxt(os.path.join(experimentDirectory, saveName + ".dat"),
                   np.transpose([q, I, err]), fmt='%1.6e', delimiter='    ')
        # make text with space between files to be opened by primus
        files.append(os.path.join(experimentDirectory, saveName + ".dat"))
    hasRun = False
    for item in files:
        if hasRun == False:
            allFiles = item
        else:
            allFiles = allFiles + " " + item  # seperate each file by a space
        hasRun = True
    if varOS.get() == "Windows":
        try:
            window.update_idletasks()
            window.update()
            subprocess.popen(os.path.join(
                entPrimusDir.get().strip(), 'primusqt.exe') + " " + allFiles)
        except:
            messagebox.showerror(title="File not found",
                                 message="Primus file not found")
    elif varOS.get() == "Linux":
        window.update_idletasks()
        window.update()
        subprocess.Popen('primus' + " " + allFiles)


def removeEntry(event=None):
    for record in tvData.selection():
        tvData.delete(record)


def repopulate():
    clearAll()
    for item in range(len(scatteringData)):
        tvData.insert(parent='', index='end', iid=scatteringData[item].params["ID"], text="", values=(scatteringData[item].params["ID"],
                                                                                                      scatteringData[item].params["file name"], round(scatteringData[item].params["sumI"], 5), round(scatteringData[item].params["TM"], 4)))


def newExperiment():
    global experimentDirectory
    experimentDirectory = askdirectory()
    lblDirectory.config(text="Directory: "+experimentDirectory)


def clearAll():
    global scatteringData
    global imageCount
    for record in tvData.get_children():
        tvData.delete(record)


def getqFromIm(event):
    global is1M, entQpos
    ix, iy = event.xdata, event.ydata
    # currently gets q! so we can just get not get ai etc
    # if is1M == True:
    #     try:
    #         ai = getAI("eiger1m")
    #     except:
    #         messagebox.showerror(title="File not found", message= "Poni2 file not found")
    #         return
    # elif is1M == False:
    #     try:
    #         ai = getAI("eiger9m") #entPoni9M.get().strip()
    #     except:
    #         messagebox.showerror(title="File not found", message= "Poni1 file not found")
    # #ai =pyFAI.load(os.path.join(experimentDirectory,poni)

    # q = ai.qFunction(iy,ix,param=None, path="cython")
    # q = np.divide(q,10) # change to nm
    q = np.sqrt(ix**2 + iy**2)
    #print(ix, iy, q)
    entQpos.delete(0, END)
    entQpos.insert(END, str(np.round(q, 5)))
    del ix, iy

def make_Eiger_mask(frame):
    # mask vales are 2^32-1 = 4294967295 for 32 bit image
    maskPixels = np.squeeze(np.where(frame == 4294967295))
    frame.fill(0)
    # for pyFAI masked values are 1 and rest are 0
    frame[maskPixels[0], maskPixels[1]] = 1
    return np.array(frame)

def apply_Eiger_mask(image):
    # mask vales are 2^32-1 = 4294967295 for 32 bit image
    maskPixels = np.squeeze(np.where(image == 4294967295))
    #maskPixels = np.squeeze(np.where(image >= 4000000000))
    image[maskPixels[0], maskPixels[1]] = 0 # for image this is zero!
    return np.array(image)

# def threshold_mask(image,threshold):
#    maskPixels = np.squeeze(np.where(image >= threshold))
#    image[maskPixels[0],maskPixels[1]] = 0
#    return np.array(image)


def showPlot2d():
    global scatteringData, experimentDirectory, plot2d, varLog2d, canvas2d, is1M, entImgMin, entImgMax, entAngle

    # CHANGE COLOR MAP HERE!!
    plt.pyplot.plasma()
    x = tvData.item(tvData.selection()[0])["values"][0]
    fileName = scatteringData[int(x)].params["fileNameSmp"]
    frame = scatteringData[int(x)].params["frame"]
    is1M = scatteringData[int(x)].params["is1M"]

    # remove mask as then we can open any image!
    if is1M == True:
        fileName = fileName[1:3] + fileName[0]
        ai = getAI("eiger1m")
        if varMask2d.get() == 1:     # first try get user mask
            try:
                mask = fabio.open(os.path.join(
                    experimentDirectory, entMask1M.get().strip())).data
            except:
                pass
                
    else:
        ai = getAI("eiger9m")
        if varMask2d.get() == 1:
            try:
                mask = fabio.open(os.path.join(
                    experimentDirectory, entMask9M.get().strip())).data
            except:
                pass
                

    if frame == 0:
        img = fabio.open(os.path.join(
            experimentDirectory, fileName+"_master.h5"))
    else:
        allFrames = fabio.open(os.path.join(
            experimentDirectory, fileName+"_master.h5"))
        img = allFrames.getframe(frame)
    plot2d.clear()

    masked_image = apply_Eiger_mask(img.data)

    # get beam center
    Fit2dDic1 = ai.getFit2D()
    BeamCenX = Fit2dDic1["centerX"]
    BeamCenY = Fit2dDic1["centerY"]
    # Mental work around need to fix
    ############################################################################
    Fit2dDic = {}
    Fit2dDic['directBeam'] = Fit2dDic1['directDist']
    Fit2dDic['beamX'] = Fit2dDic1['centerX']
    Fit2dDic['beamY'] = Fit2dDic1['centerY']
    Fit2dDic['tilt'] = 0
    Fit2dDic['tiltPlanRotation'] = 0
    Fit2dDic['detector'] = ai.get_config()['detector']
    plankC = float(4.135667696e-15)  # Planck's constant in ev/Hz
    speedLightC = float(299_792_458)  # speed of light m/s
    Fit2dDic["energy"] = plankC * speedLightC / 1000 / ai.get_wavelength()
    #dic = ai.get_config()
    #ai = pyFAI.geometry.Geometry.set_config(**dic)
    #############################################################################
    del ai
    ai = makeAIFit2d(**Fit2dDic)  # mental workaround which might work??
    # print(np.dtype(BeamCenX))

    # get the q positions for image
    imageSize = masked_image.shape
    imageX = imageSize[1]
    imageY = imageSize[0]
    #print("Only works with pSAXSpar.txt and WAXSpar.txt need to fix to work with poni")
    #qZ1 = ai.qFunction(1,1,param=None, path="cython")
    qZ0 = ai.qFunction(imageY-1, BeamCenX, param=None, path="cython")
    qZ0 = np.squeeze(np.divide(qZ0, 10))  # change to nm

    qZ1 = ai.qFunction(0, BeamCenX, param=None, path="cython")
    qZ1 = np.squeeze(np.divide(qZ1, 10))  # change to nm

    qX0 = ai.qFunction(BeamCenY, 0, param=None, path="cython")
    qX0 = np.squeeze(np.divide(qX0, 10))  # change to nm

    qX1 = ai.qFunction(BeamCenY, imageX-1, param=None, path="cython")
    qX1 = np.squeeze(np.divide(qX1, 10))  # change to nm

    if BeamCenX > 0:  # define below zero
        qX0 = -qX0

    if BeamCenY > 0:
        qZ0 = -qZ0

    if varMask2d.get() == 1:
        try:
            mask = 1-mask  # invert mask
            masked_image = np.multiply(masked_image, mask)
        except:
            pass
            #messagebox.showwarning(title='Warning', message='No user Mask supplied')

    if var2dThreshold.get() == 1:
        minColor = float(entImgMin.get().strip())
        maxColor = float(entImgMax.get().strip())
    elif var2dThreshold.get() == 0:
        #minColor = np.amin(masked_image)
        minColor = 0
        maxColor = np.amax(masked_image)
    if varRotate.get() == 1:
        masked_image = IImage.fromarray(masked_image)
        masked_image = masked_image.rotate(float(entAngle.get().strip()), center=(
            int(BeamCenX), int(BeamCenY)), resample=IImage.BICUBIC)
        masked_image = np.asarray(masked_image)

    if varLog2d.get() == 1:
        # plot2d.imshow(np.log10(np.multiply(img.data,mask)))
        if minColor > 0:
            minColor = np.log10(minColor)
        plot2d.imshow(np.log10(masked_image), vmin=minColor, vmax=np.log10(
            maxColor), extent=[qX0, qX1, qZ0, qZ1])  # log 0 problem
    else:
        # plot2d.imshow(np.multiply(img.data,mask))
        plot2d.imshow(np.array(masked_image), vmin=minColor,
                      vmax=maxColor, extent=[qX0, qX1, qZ0, qZ1])

    plot2d.set_xlabel(r'q_x [$\AA$]')
    plot2d.set_ylabel(r'q_z [$\AA$]')
    plot2d.tick_params(labelsize=8)
    # plt.pyplot.plasma()
    fig2d.tight_layout()
    canvas2d.draw()


def show2dWindow():
    global canvas2d, plot2d, varLog2d, fig2d, entQpos, entImgMin, entImgMax, entAngle
    # other window
    window2d = Toplevel(window)
    # Define title for window
    window2d.title("2d image viewing")
    # specify size
    window2d.geometry("650x700")
    framePlot2d = Frame(window2d, width=600, height=600)
    framePlot2d.propagate(0)
    framePlot2d.grid(column=0, row=2, columnspan=20, rowspan=20)
    fig2d = Figure(dpi=100)
    fig2d.set_size_inches(5.3, 5.3, forward=True)
    plot2d = fig2d.add_subplot(111)
    canvas2d = FigureCanvasTkAgg(fig2d, master=framePlot2d)
    canvas2d.draw()
    canvas2d.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    toolbar2d = NavigationToolbar2Tk(canvas2d, framePlot2d)
    toolbar2d.update()
    canvas2d.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    # Create Gui stuff

    button1 = Button(window2d, text="Exit", command=window2d.destroy)
    button2 = Button(window2d, text="Show selected image", command=showPlot2d)
    cbRotate = Checkbutton(window2d, text="Rotate: Ang.",
                           variable=varRotate, onvalue=1, offvalue=0)
    entAngle = Entry(window2d, width=4)
    cbLog2d = Checkbutton(window2d, text="Log intensity",
                          variable=varLog2d, onvalue=1, offvalue=0)
    cbLog2d.select()
    cbApplyMask = Checkbutton(
        window2d, text="Apply mask", variable=varMask2d, onvalue=1, offvalue=0)
    cbApplyMask.select()
    cbApplyThreshold = Checkbutton(
        window2d, text="Apply threshold", variable=var2dThreshold, onvalue=1, offvalue=0)
    lblClickedq = Label(window2d, text="Clicked q [Ang.]:")
    entQpos = Entry(window2d, width=8)
    entImgMax = Entry(window2d, width=8)
    entImgMax.insert(END, 4294967295)
    entImgMin = Entry(window2d, width=8)
    entImgMin.insert(END, 0)
    lblImgMax = Label(window2d, text="Map max:")
    lblImgMin = Label(window2d, text="Map min:")
    button1.grid(column=0, row=0)
    button2.grid(column=1, row=0)
    cbLog2d.grid(column=2, row=0)
    lblClickedq.grid(column=3, row=0)
    entQpos.grid(column=4, row=0)
    cbApplyMask.grid(column=5, row=0)
    lblImgMax.grid(column=0, row=1)
    entImgMax.grid(column=1, row=1, sticky='w')
    lblImgMin.grid(column=1, row=1, sticky='e')
    entImgMin.grid(column=2, row=1, sticky='w')
    cbApplyThreshold.grid(column=3, row=1, sticky='w')
    cbRotate.grid(column=4, row=1, sticky='w')
    entAngle.grid(column=5, row=1, sticky='w')
    cid = canvas2d.mpl_connect('button_press_event', getqFromIm)

    def onsize(event):
        window_width = window2d.winfo_width()
        window_height = window2d.winfo_height()
        framePlot2d.config(width=window_width-50, height=window_height - 50)
        fig2d.set_size_inches(
            (window_width-50)/100, (window_height - 100)/100, forward=True)  # 100 dpi
    window2d.bind("<Configure>", onsize)


def handle_tab_changed(event):
    selection = event.widget.select()
    tab = event.widget.tab(selection, "text")
    if tab == "Data Reduction":
        window.unbind('q')
        window.unbind('w')
        window.unbind('e')
        window.unbind('r')
        window.unbind('s')
        window.unbind("<space>")
        window.unbind('m')

    if tab == "Plotting and Averaging":
        window.bind('q', plotSelectedData)
        window.bind('w', averageSelected)
        window.bind('e', saveSelection)
        window.bind('r', removeEntry)
        window.bind('s', subBackground)
        window.bind('m', merge)
        window.bind("<space>", selectBackground)


def showHelp():
    helpWindow = Toplevel(window)
    # helpWindow.geometry("400x400")
    helpWindow.title("Help and info")
    Label(helpWindow, font=('TkDefaultFont', 15),
          text="This is currently not fully tested software, please carefully check any results. \nhttps://sourceforge.net/projects/weicomplex/support for help or to report bugs").pack()

# def do_popup(event):
#    try:
#        menuPlotting.tk_popup(event.x_window, event.y_window)
#    finally:
#        menuPlotting.grab_release()


def clearMemory():
    global scatteringData
    try:
        scatteringData = []
        messagebox.showinfo(title="Memory Cleared",
                            message="Scattering data removed from memory.")
    except:
        messagebox.showinfo(
            title="No data.", message="No scattering data to clear.")


def loadExperiment():
    global experimentDirectory
    try:
        scatteringDataTemp = np.load(open(os.path.join(
            experimentDirectory, entExperimentFile.get().strip()), 'rb'), allow_pickle=True)
        for item in scatteringDataTemp:
            scatteringData.append(item)
            #scatteringData.append([scatteringDataTemp[item][0], scatteringDataTemp[item][1], scatteringDataTemp[item][2], scatteringDataTemp[item][3], scatteringDataTemp[item][4], scatteringDataTemp[item][5], scatteringDataTemp[item][6], scatteringDataTemp[item][7],scatteringDataTemp[item][8],scatteringDataTemp[item][9],scatteringDataTemp[item][10]])
            tvData.insert(parent='', index='end', iid=scatteringData[item].params["ID"], text="", values=(scatteringData[item].params["ID"],
                                                                                                          scatteringData[item].params["file name"], round(scatteringData[item].params["sumI"], 5), round(scatteringData[item].params["TM"], 4)))
        messagebox.showinfo(title="Data file read into memory", message=str(
            len(scatteringData)) + " 1d profiles loaded")
    except:
        messagebox.showinfo(title="Data file not found",
                            message="No file, 0 profiles loaded")


def incrementEntryRow():
    global entTransAirFrm, entBkgSAXSFrm, entTransBkgFrm, entTransSmpFrm, entSmpSAXSFrm, varSubBkg

    startRow = entTransSmpFrm[-1].grid_info()['row']

    varSubBkg.append(IntVar())

    entTransAirFrm.append(Entry(frameReduction, width=5))
    entTransSmpFrm.append(Entry(frameReduction, width=5))
    entSmpSAXSFrm.append(Entry(frameReduction, width=5))
    entTransBkgFrm.append(Entry(frameReduction, width=5))
    entBkgSAXSFrm.append(Entry(frameReduction, width=5))
    cbSubBkg.append(Checkbutton(frameReduction, text="Sub. Bkg",
                    variable=varSubBkg[-1], onvalue=1, offvalue=0))
    cbSubBkg[-1].select()
    entTransAirFrm[-1].grid(column=3, row=startRow + 1)
    entTransSmpFrm[-1].grid(column=4, row=startRow + 1)
    entSmpSAXSFrm[-1].grid(column=5, row=startRow + 1)
    entTransBkgFrm[-1].grid(column=6, row=startRow + 1)
    entBkgSAXSFrm[-1].grid(column=7, row=startRow + 1)
    cbSubBkg[-1].grid(column=8, row=startRow + 1)


def find_overlap(q_data9M, q_data1M):

    overlap_1M_indices = []
    overlap_9M_indices = []
    for i_q1M, v_q1M in enumerate(q_data1M):
        if float(v_q1M) < float(q_data9M[-1]):
            overlap_1M_indices.append(i_q1M)

    for i_q9M, v_q9M in enumerate(q_data9M):
        if v_q9M >= q_data1M[1]:
            overlap_9M_indices.append(i_q9M)
    return overlap_1M_indices, overlap_9M_indices


def calc_scale_factor(data9M, data1M, overlap_1M_indices, overlap_9M_indices):
    Idat_9M = np.array(data9M.I)
    Idat_1M = np.array(data1M.I)
    scale_factor = np.divide(
        np.mean(Idat_9M[overlap_9M_indices]), np.mean(Idat_1M[overlap_1M_indices]))
    return scale_factor


def merge(event=None):
    global scatteringData
    # need to put a key in the dictionary for merged data tomorrow then this will work
    # will only put merged in here

    # be very careful we are getting indeces from the treeview and using with the total data matrix
    if len(tvData.selection()) != 2:
        messagebox.showerror('Error', 'Select 2 frames from the list')
        return

    flag1M = []
    ID = []
    for record in tvData.selection():
        x = tvData.item(record)["values"][0]
        ID.append(x)
        flag1M.append(scatteringData[int(x)].params["is1M"])

    if (flag1M[0] == flag1M[1]):
        messagebox.showerror(
            'Error', 'Only 1 1M and 1 9M frame should be selected.')
        return

    # get 9M and 1M ID

    if flag1M[0]:
        ID1M = ID[0]
        ID9M = ID[1]
    else:
        ID1M = ID[1]
        ID9M = ID[0]

    start_index_9M = int(ent9MStart.get().strip())
    end_index_9M = int(ent9MEnd.get().strip())
    start_index_1M = int(ent1MStart.get().strip())
    end_index_1M = int(ent1MEnd.get().strip())

    if varAutoScale.get() == 1:
        overlap_1M_indices, overlap_9M_indices = find_overlap(scatteringData[ID9M].q[start_index_9M:end_index_9M],
                                                              scatteringData[ID1M].q[start_index_1M:end_index_1M])
        scale = calc_scale_factor(
            scatteringData[ID9M], scatteringData[ID1M], overlap_1M_indices, overlap_9M_indices)
        entScale1M.delete(0, END)
        entScale1M.insert(END, str(round(scale, 8)))
    elif varAutoScale.get() == 0:
        scale = float(entScale1M.get().strip())
    new_data_q = np.append(np.array(scatteringData[ID9M].q[start_index_9M:end_index_9M]), np.array(
        scatteringData[ID1M].q[start_index_1M:end_index_1M]))
    new_data_I = np.append(np.array(scatteringData[ID9M].I[start_index_9M:end_index_9M]), np.multiply(
        scatteringData[ID1M].I[start_index_1M:end_index_1M], scale))
    new_data_err = np.append(np.array(scatteringData[ID9M].err[start_index_9M:end_index_9M]), np.multiply(
        scatteringData[ID1M].err[start_index_1M:end_index_1M], scale))
    sort_index = np.argsort(new_data_q)
    new_data_q = new_data_q[sort_index]
    new_data_I = new_data_I[sort_index]
    new_data_err = new_data_err[sort_index]

    params = {"file name": scatteringData[ID9M].params["file name"]+"_merge", "ID": scatteringData[-1].params["ID"]+1, "multiDet": True, "sumI": np.sum(new_data_I),
              "TM": 1.0, "is1M": False, "fileNameSmp": scatteringData[ID9M].params["fileNameSmp"], "frame": scatteringData[ID9M].params["frame"], "isMerge": True}
    scatteringData.append(
        expData(new_data_q, new_data_I, new_data_err, params))

    try:
        np.array(scatteringData, dtype=object).dump(
            open(os.path.join(experimentDirectory, entExperimentFile.get()), 'wb'))
        # tvData.insert(parent='',index='end',iid=scatteringData[-1][1], text="", values=(scatteringData[-1][1],scatteringData[-1][0],round(scatteringData[-1][3],5),scatteringData[-1][7])) # update treeview data
        tvData.insert(parent='', index='end', iid=scatteringData[-1].params["ID"], text="", values=(scatteringData[-1].params["ID"],
                                                                                                    scatteringData[-1].params["file name"], round(scatteringData[-1].params["sumI"], 5), round(scatteringData[-1].params["TM"], 4)))
    except:
        messagebox.showinfo(
            title="Error", message="Save data location unknown, \nplease select file > save experiment \n or check advanced tab has the data file with .wc \nNo data processing performed")


def toggle_entry_boxes():
    if varCalcTMbkg.get() == 1:
        entbkgTMVal.config(state='disabled')
    elif varCalcTMbkg.get() == 0:
        entbkgTMVal.config(state='normal')

    if varCalcTMSmp.get() == 1:
        entsmpTMVal.config(state='disabled')
    elif varCalcTMSmp.get() == 0:
        entsmpTMVal.config(state='normal')

    if varPsaxspar.get() == 1:
        entPoni9M.config(state='disabled')
    elif varPsaxspar.get() == 0:
        entPoni9M.config(state='normal')

    if varWaxspar.get() == 1:
        entPoni1M.config(state='disabled')
    elif varWaxspar.get() == 0:
        entPoni1M.config(state='normal')
    # if varSubBkg.get() == 1:
    #    entBkgSAXSFrm.config(state='disabled')
    #    entTransBkgFrm.config(state='disabled') # todo


def onsize(event):
    window_width = window.winfo_width()
    window_height = window.winfo_height()
    #print(window_width, window_height)
    if window_width > 1200:
        framePlot.config(width=window_width-730, height=window_height - 80)
        fig.set_size_inches((window_width-730)/100,
                            (window_height-110)/100, forward=True)  # 100 dpi
    else:
        framePlot.config(width=500, height=580)
        fig.set_size_inches(500/100, 520/100, forward=True)  # 100 dpi


##################################################
# Initialization stuff
##################################################
global scatteringData, backGroundID, window2dOpen, fig2d, experimentDirectory
global graph_past
window2dOpen = False
scatteringData = None  # initialize variable global var which will contain all our data
graph_past = False


############################################################################################################################################################################
# GUI Below
#############################################################################################################################################################################
# menu bar
menubar = Menu(window)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New Experiment (set dir)", command=newExperiment)
filemenu.add_command(label="Load Experiment File", command=loadParams)
filemenu.add_command(label="Save Experiment File", command=saveParams)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=window.quit)
menubar.add_cascade(label="File", menu=filemenu)

calibmenu = Menu(menubar, tearoff=0)
calibmenu.add_command(label="PONI <> FIT2D operations",
                      command=openPONIfileWindow)
calibmenu.add_command(label="Calibrate Detector",
                      command=call_pyFAI_calibration)
calibmenu.add_command(label="Create Mask", command=mask_image)
menubar.add_cascade(label="Calibration", menu=calibmenu)

viewmenu = Menu(menubar, tearoff=0)
viewmenu.add_command(label="(Re)open image window", command=show2dWindow)
menubar.add_cascade(label="View", menu=viewmenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help", command=showHelp)
helpmenu.add_command(label="About...", command=showHelp)
menubar.add_cascade(label="Help", menu=helpmenu)


# Tabs
mainNotebook = ttk.Notebook(window)
mainNotebook.pack(fill="both", expand=1, pady=15, padx=15)
frameReduction = Frame(mainNotebook)  # ,width = 1100, height = 750)
framePlotting = Frame(mainNotebook)  # ,width = 1100, height = 750)
framePlotting.pack(fill="both", expand=1)
canvasPlotting = Canvas(framePlotting)

scrollPlottingY = ttk.Scrollbar(
    framePlotting, orient=VERTICAL, command=canvasPlotting.yview)
scrollPlottingY.pack(side=RIGHT, fill=BOTH)
scrollPlottingX = ttk.Scrollbar(
    framePlotting, orient=HORIZONTAL, command=canvasPlotting.xview)
scrollPlottingX.pack(side=BOTTOM, fill=X)
# canvasPlotting.config(yscrollcommand=scrollPlottingY.set)
canvasPlotting.config(xscrollcommand=scrollPlottingX.set)
canvasPlotting.bind('<Configure>', lambda e: canvasPlotting.configure(
    scrollregion=canvasPlotting.bbox("all")))
framePlotting2 = Frame(canvasPlotting)
canvasPlotting.create_window((0, 0), window=framePlotting2, anchor="nw")
canvasPlotting.pack(side=LEFT, fill=BOTH, expand=1)
frameGISAXS = Frame(mainNotebook)  # ,width = 1100, height = 750)
frameAdvanced = Frame(mainNotebook)  # , width = 1100, height = 750)
frameReduction.pack(fill="both", expand=1)

frameGISAXS.pack(fill="both", expand=1)
frameAdvanced.pack(fill="both", expand=1)
mainNotebook.add(frameReduction, text="Data Reduction")
mainNotebook.add(framePlotting, text="Plotting and Averaging")
mainNotebook.add(frameGISAXS, text="GISAXS / GIWAXS")
mainNotebook.add(frameAdvanced, text="Advanced Options")


# text and boxes
#entBeamX = Entry(frameReduction, width = 10)
#entBeamY = Entry(frameReduction, width = 10)
entsmpTMVal = Entry(frameReduction, width=20)
entbkgTMVal = Entry(frameReduction, width=20)
entMask9M = Entry(frameReduction, width=20)  # 9M
entMask1M = Entry(frameReduction, width=20)  # 1M
entPoni9M = Entry(frameReduction, width=20)
entPoni1M = Entry(frameReduction, width=20)
entScaleFactor = Entry(frameReduction, width=20)
entThickness = Entry(frameReduction, width=20)
entNumPoints = Entry(frameReduction, width=20)

# need to make these into a list

entTransAirFrm = []
entTransBkgFrm = []
entBkgSAXSFrm = []
entTransSmpFrm = []
entSmpSAXSFrm = []

entTransAirFrm.append(Entry(frameReduction, width=5))
entTransSmpFrm.append(Entry(frameReduction, width=5))
entSmpSAXSFrm.append(Entry(frameReduction, width=5))
entTransBkgFrm.append(Entry(frameReduction, width=5))
entBkgSAXSFrm.append(Entry(frameReduction, width=5))

# frame plotting
entScaleSmp = Entry(framePlotting2, width=7)
entScaleSmp.insert(END, "1.0")
entScaleBkg = Entry(framePlotting2, width=7)
entScaleBkg.insert(END, "1.0")
entLineWidth = Entry(framePlotting2, width=5)
entLineWidth.insert(END, "0.5")
ent1MStart = Entry(framePlotting2, width=5)
ent1MStart.insert(END, "0")
ent1MEnd = Entry(framePlotting2, width=5)
#ent1MEnd.insert(txtNumPoints.get("1.0", 'end-1c'))
ent9MStart = Entry(framePlotting2, width=5)
ent9MStart.insert(END, "0")
ent9MEnd = Entry(framePlotting2, width=5)
entScale1M = Entry(framePlotting2, width=10)
entScale1M.insert(END, "1.0")
#ent9MEnd.insert(txtNumPoints.get("1.0", 'end-1c'))
#txtScale1M = Text(framePlotting2, height=1, width=20)
# txtScale1M.insert(END,"0.001")
# frame advanced
entAnacondaDir = Entry(frameAdvanced, width=40)
entAnacondaDir.insert(END, "C:/Users/user/anaconda3/Scripts")
entPrimusDir = Entry(frameAdvanced, width=40)
entPrimusDir.insert(END, "C:/some/dir/where/it/exists")
entExperimentFile = Entry(frameAdvanced, width=20)
entExperimentFile.insert(END, "experiment.wc")


# Check box
varCalcTMSmp = IntVar()
varCalcTMbkg = IntVar()
varSubBkg = []
varSubBkg.append(IntVar())
varLogX = IntVar()
varLogY = IntVar()
varErrorBar = IntVar()
varGrid = IntVar()
varLog2d = IntVar()
var2dThreshold = IntVar()
varMask2d = IntVar()
varAutoScale = IntVar()
varPsaxspar = IntVar()
varWaxspar = IntVar()
varExportas = IntVar()
varRotate = IntVar()
varEigerMask = IntVar()
varReject = IntVar()

cbManTMSmp = Checkbutton(frameReduction, text="Calc TM", variable=varCalcTMSmp,
                         onvalue=1, offvalue=0, command=toggle_entry_boxes)
cbManTMSmp.select()
cbManTMBkg = Checkbutton(frameReduction, text="Calc TM", variable=varCalcTMbkg,
                         onvalue=1, offvalue=0, command=toggle_entry_boxes)
cbManTMBkg.select()
cbSubBkg = []
cbSubBkg.append(Checkbutton(frameReduction, text="Sub. Bkg",
                variable=varSubBkg[-1], onvalue=1, offvalue=0))
cbSubBkg[-1].select()
cbPsaxsPar = Checkbutton(frameReduction, text="pSAXSpar.txt",
                         variable=varPsaxspar, onvalue=1, offvalue=0, command=toggle_entry_boxes)
cbwaxsPar = Checkbutton(frameReduction, text="WAXSpar.txt",
                        variable=varWaxspar, onvalue=1, offvalue=0, command=toggle_entry_boxes)
cbRejectMask = Checkbutton(frameReduction, text="REJECT.dat",
                        variable=varReject, onvalue=1, offvalue=0)

# frame Plotting
cbLogX = Checkbutton(framePlotting2, text="Log X",
                     variable=varLogX, onvalue=1, offvalue=0)
cbLogX.select()
cbLogY = Checkbutton(framePlotting2, text="Log Y",
                     variable=varLogY, onvalue=1, offvalue=0)
cbLogY.select()
cbErrorBar = Checkbutton(framePlotting2, text="Err. Bars",
                         variable=varErrorBar, onvalue=1, offvalue=0, padx=2)
cbErrorBar.select()
cbGrid = Checkbutton(framePlotting2, text="Grid",
                     variable=varGrid, onvalue=1, offvalue=0)
cbAutoScale = Checkbutton(framePlotting2, text="Auto\nscale",
                          variable=varAutoScale, onvalue=1, offvalue=0)
cbExportas = Checkbutton(framePlotting2, text="Exp. as",
                         variable=varExportas, onvalue=1, offvalue=0)

# advanced options
cbUseEigerMask = Checkbutton(frameAdvanced, text="Use Eiger Mask", variable=varEigerMask, onvalue=1, offvalue=0)
cbUseEigerMask.select()

# Progress bars
progBarSAXS = ttk.Progressbar(
    frameReduction, orient=HORIZONTAL, length=300, mode='determinate')
progBarWAXS = ttk.Progressbar(
    frameReduction, orient=HORIZONTAL, length=300, mode='determinate')

# options menu or dropdown menu
varBeamPixels = StringVar()
varBeamPixels.set("Beam 5 pixels")
varOS = StringVar()
varOS.set("Windows")
varDetector = StringVar()
varDetector.set("9M and 1M")
varOperations = StringVar()
varOperations.set("9M and 1M")
varSave = StringVar()
varSave.set("Save All Data")
varPlotStyle = StringVar()
varPlotStyle.set("Line")

omBeamPixels = OptionMenu(frameReduction, varBeamPixels,
                          "Beam 3 pixels", "Beam 5 pixels", "Beam 7 pixels")
omDetector = OptionMenu(frameReduction, varDetector, "9M and 1M", "9M", "1M")
omSave = OptionMenu(frameReduction, varSave, "Save All Data", "Manual Save")
omOperate = OptionMenu(framePlotting2, varOperations, "9M and 1M", "Seperate")

omPlotStyle = OptionMenu(framePlotting2, varPlotStyle, "Line", "Points")

omOS = OptionMenu(frameAdvanced, varOS, "Windows", "Linux")
# Main treeview
frame_tvData = LabelFrame(
    framePlotting2, text="All Data", font=('TkDefaultFont', 15))
frame_tvData.grid(column=0, row=0, columnspan=3, rowspan=8, pady=5, padx=5)
# scroll bar
scroll_tvData = Scrollbar(frame_tvData)
scroll_tvData.pack(side=RIGHT, fill=Y)
tvData = ttk.Treeview(
    frame_tvData, yscrollcommand=scroll_tvData.set, selectmode="extended", height=15)
scroll_tvData.config(command=tvData.yview)
tvData['columns'] = ("Id", "Name", "sum(I)", "TM")
tvData.column("#0", width=5, minwidth=5)
tvData.column("Id", width=40, minwidth=25, anchor=CENTER)
tvData.column("Name", anchor=W, width=120)
tvData.column("sum(I)", anchor=CENTER, width=80)
tvData.column("TM", anchor=CENTER, width=80)

tvData.heading("#0", text="", anchor=W)
tvData.heading("Id", text="Id")
tvData.heading("Name", text="Name")
tvData.heading("sum(I)", text="sum(I)")
tvData.heading("TM", text="TM")

# labels
lblDirectory = Label(frameReduction, text="Directory: Please select")
#lblBeamX = Label(frameReduction,text="X Beam Center:")
#lblBeamY = Label(frameReduction,text="Y Beam Center:")
lblTMVal = Label(frameReduction, text="TM Sample: ")
lblTMBkgVal = Label(frameReduction, text="TM Background: ")
lblTMAirFrm = Label(frameReduction, text="TM \n Frame Air")
lblTMSmpFrm = Label(frameReduction, text="TM \n Frame sample")
lblSAXSsmpFrm = Label(frameReduction, text="SAXS \n Frames Sample")
lblTMBkgFrm = Label(frameReduction, text="TM \n Frame Background")
lblSAXSBkgFrm = Label(frameReduction, text="SAXS \n Frame Background")
lblScaleFactor = Label(frameReduction, text="Scaling Factor:")
lblMask9M = Label(frameReduction, text="9M Mask File Name:")
lblMask1M = Label(frameReduction, text="1M Mask File Name:")
lblPoni9M = Label(frameReduction, text="9M PONI File Name:")
lblPoni1M = Label(frameReduction, text="1M PONI File Name:")
lblThickness = Label(frameReduction, text="Thickness (mm)")
lblNumPoints = Label(frameReduction, text="Number of Points")
lblProgBarSAXS = Label(frameReduction, text="SAXS Reduction Progress:")
lblProgBarWAXS = Label(frameReduction, text="WAXS Reduction Progress:")


# framePlotting
framePlot = Frame(framePlotting2, width=600, height=600)
framePlot.propagate(0)
lblBackground = Label(
    framePlotting2, text="Background: Please Select", font=('TkDefaultFont', 12))
lblScaleNotice = Label(
    framePlotting2, text="When Subtracting:", font=('TkDefaultFont', 15))
lblScaleSmp = Label(framePlotting2, text="Scale Smp:")
lblScaleBkg = Label(framePlotting2, text="Scale Bkg:")
lblOperations = Label(framePlotting2, text="Perform operations on:")
lblScale1M = Label(framePlotting2, text="Scale 1M \n (on figure)")
#lblLineWidth = Label(framePlotting2,text="Line width:")
lbl9MplotStartEnd = Label(framePlotting2, text="< 9M <")
lbl1MplotStartEnd = Label(framePlotting2, text="< 1M <")
#lblTo1M = Label(framePlotting2,text="To")
#lblTo9M = Label(framePlotting2,text="To")
lblScale1M = Label(framePlotting2, text="Scale 1M \n(plotting/merge)")
lblstatus = Label(framePlotting2, text="Status:")

# Frame advanced
lblOS = Label(frameAdvanced, text="Operating System:")
lblAnacondaDir = Label(
    frameAdvanced, text="Anaconda script dir :\n (win OS only)")
lblPrimusDir = Label(frameAdvanced, text="Primus dir :\n (win OS only)")
lblExperimentFile = Label(frameAdvanced, text="Data File Name")


# Buttons for calibration and intial image viewing ############################################

btn_get_tansmission = Button(
    frameReduction, text="Get transmission:", command=clickGetTransmission)
btn_integrate_SAXS = Button(frameReduction, text="RUN:", command=integrateMultiDet, font=(
    'TkDefaultFont', 20), bg='green', fg='white')
btn_increase_entry_row = Button(
    frameReduction, text="Increase row", command=incrementEntryRow)

# plotting
btn_plot_selected = Button(
    framePlotting2, text="Plot Selected\n(q)", command=plotSelectedData)
btn_av_selection = Button(
    framePlotting2, text="Average Selection\n(w)", command=averageSelected)
btn_select_bkg = Button(
    framePlotting2, text="Is Background \n(spacebar)", command=selectBackground)
btn_remove_entry = Button(
    framePlotting2, text="(R)emove Selection", command=removeEntry)
btn_subtract = Button(
    framePlotting2, text="(S)ubtract Background \n From Selected Frames", command=subBackground, bg="cyan")
btn_save_selection = Button(
    framePlotting2, text="(E)xport Selection", command=saveSelection)
btn_clear_all = Button(framePlotting2, text="Clear All", command=clearAll, font=(
    'TkDefaultFont', 12), bg='red', fg='white')
btn_repopulate = Button(framePlotting2, text="Repopulate", command=repopulate, font=(
    'TkDefaultFont', 12), bg='green', fg='white')
btn_primus = Button(
    framePlotting2, text="Primus Merge Selected", command=primusOpen)
btn_2d_show_selected = Button(
    framePlotting2, text="Show in 2d in viewer", command=showPlot2d, bg="orange")
btn_merge = Button(framePlotting2, text="(M)erge",
                   command=merge, font=('TkDefaultFont', 12), bg='snow')
# Frame advanced
btn_clear_memory = Button(
    frameAdvanced, text="Clear memory", command=clearMemory)
btn_load_experiment = Button(
    frameAdvanced, text="Load experiment data", command=loadExperiment)
###############
# menu
menuPlotting = Menu(frameReduction, tearoff=0)
menuPlotting.add_command(label="log-log")
menuPlotting.add_command(label="lin-log")
###############################################################################################################################################################
# put everything on the GUI
################################################################################################################################################################
#btn_calibration.grid(column=0, row=0)
# btn_mask_image.grid(column=0,row=2)
# btn_cen_air_frame.grid(column=0,row=3)
# btn_save_params.grid(column=1,row=1)
# btn_load_params.grid(column=1,row=2)
btn_integrate_SAXS.grid(column=0, row=1, rowspan=2)
omBeamPixels.grid(column=1, row=0)
btn_get_tansmission.grid(column=0, row=0)
omDetector.grid(column=1, row=1)
omSave.grid(column=1, row=2)
lblTMAirFrm.grid(column=3, row=0)
lblTMSmpFrm.grid(column=4, row=0)
lblSAXSsmpFrm.grid(column=5, row=0)
lblTMBkgFrm.grid(column=6, row=0)
lblSAXSBkgFrm.grid(column=7, row=0)
entTransAirFrm[0].grid(column=3, row=1)
entTransSmpFrm[0].grid(column=4, row=1)
entSmpSAXSFrm[0].grid(column=5, row=1)
entTransBkgFrm[0].grid(column=6, row=1)
entBkgSAXSFrm[0].grid(column=7, row=1)
cbSubBkg[-1].grid(column=8, row=1)
btn_increase_entry_row.grid(column=8, row=0)


# lblBeamX.grid(column=0,row=3)
# lblBeamY.grid(column=0,row=4)
lblTMVal.grid(column=0, row=3)
lblTMBkgVal.grid(column=0, row=4)
entbkgTMVal.grid(column=1, row=4)
# entBeamX.grid(column=1,row=3)
# entBeamY.grid(column=1,row=4)
entsmpTMVal.grid(column=1, row=3)
cbManTMSmp.grid(column=2, row=3, sticky='w')
cbManTMBkg.grid(column=2, row=4, sticky='w')
entMask9M.grid(column=1, row=5)
lblMask9M.grid(column=0, row=5)
cbRejectMask.grid(column=2, row=5, sticky='w')
entMask1M.grid(column=1, row=6)
lblMask1M.grid(column=0, row=6)
entPoni9M.grid(column=1, row=7)
lblPoni9M.grid(column=0, row=7)
cbPsaxsPar.grid(column=2, row=7)
entPoni1M.grid(column=1, row=8)
lblPoni1M.grid(column=0, row=8)
cbwaxsPar.grid(column=2, row=8)
lblScaleFactor.grid(column=0, row=9)
entScaleFactor.grid(column=1, row=9)
lblThickness.grid(column=0, row=10)
entThickness.grid(column=1, row=10)
lblNumPoints.grid(column=0, row=11)
entNumPoints.grid(column=1, row=11)
lblProgBarSAXS.grid(column=1, row=12)
progBarSAXS.grid(column=0, row=13, columnspan=3)
lblProgBarWAXS.grid(column=1, row=14)
progBarWAXS.grid(column=0, row=15, columnspan=3)
lblDirectory.grid(column=0, row=16, columnspan=3, sticky='w')

# frame plotting
tvData.pack()
btn_plot_selected.grid(column=3, row=3)
btn_av_selection.grid(column=3, row=4)
btn_save_selection.grid(column=3, row=5)
cbExportas.grid(column=3, row=5, sticky='s')
btn_remove_entry.grid(column=3, row=6)
btn_select_bkg.grid(column=3, row=7)
btn_subtract.grid(column=2, row=10)
cbErrorBar.grid(column=3, row=10, sticky='w')
cbGrid.grid(column=3, row=10, sticky='e')
btn_primus.grid(column=1, row=10)
btn_2d_show_selected.grid(column=0, row=10)
lblBackground.grid(column=0, row=9, columnspan=4)
lblScaleNotice.grid(column=0, row=12, columnspan=2, sticky='w')
lblScaleSmp.grid(column=0, row=13, sticky='w')
entScaleSmp.grid(column=0, row=13, sticky='e')
lblScaleBkg.grid(column=0, row=14, sticky='w')
entScaleBkg.grid(column=0, row=14, sticky='e')
# lblScale1M.grid(column=0,row=8)
# txtScale1M.grid(column=1,row=8)
btn_clear_all.grid(column=1, row=11)
btn_repopulate.grid(column=0, row=11)
# lblLineWidth.grid(column=3,row=11,sticky='w')
omPlotStyle.grid(column=3, row=11, sticky='w')
entLineWidth.grid(column=3, row=11, sticky='e')
lbl1MplotStartEnd.grid(column=2, row=11)
ent1MStart.grid(column=2, row=11, sticky='w')
#lblTo1M.grid(column=2,row=11,sticky = 's')
ent1MEnd.grid(column=2, row=11, sticky='e')
lbl9MplotStartEnd.grid(column=2, row=12)
ent9MStart.grid(column=2, row=12, sticky='w')
#lblTo9M.grid(column=2,row=12,sticky = 's')
ent9MEnd.grid(column=2, row=12, sticky='e')
cbLogX.grid(column=3, row=12, sticky='e')
cbLogY.grid(column=3, row=12, sticky='w')
lblScale1M.grid(column=2, row=13, sticky='e')
entScale1M.grid(column=3, row=13, sticky='w')
btn_merge.grid(column=1, row=13, sticky='e')
cbAutoScale.grid(column=3, row=13, sticky='e')
lblstatus.grid(column=1, row=14, columnspan=3)

# Frame advanced options
lblOS.grid(column=0, row=0)
omOS.grid(column=1, row=0, sticky='w')
lblAnacondaDir.grid(column=0, row=1)
entAnacondaDir.grid(column=1, row=1, sticky='w')
lblPrimusDir.grid(column=0, row=2)
entPrimusDir.grid(column=1, row=2, sticky='w')
lblExperimentFile.grid(column=0, row=3)
entExperimentFile.grid(column=1, row=3, sticky='w')
btn_clear_memory.grid(column=0, row=4)
btn_load_experiment.grid(column=0, row=5)
cbUseEigerMask.grid(column=0, row=6)

# plotting data
framePlot.grid(column=8, row=3, columnspan=20, rowspan=20, padx=10)
fig = Figure(figsize=(5, 5.3), dpi=100)
plot1 = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=framePlot)
# canvas.draw()

toolbar = NavigationToolbar2Tk(canvas, framePlot)
toolbar.update()
#canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=0)
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
# Bind the right click

toggle_entry_boxes()

# key binding
mainNotebook.bind("<<NotebookTabChanged>>", handle_tab_changed)
window.config(menu=menubar)
window.bind("<Configure>", onsize)
# Call click func

show2dWindow()  # open window at initialization

window.mainloop()
