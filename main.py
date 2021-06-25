# This Python file uses the following encoding: utf-8
import os, sys, io
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
from skimage.color import rgb2hed, hed2rgb, separate_stains, combine_stains
from skimage.exposure import rescale_intensity
from matplotlib.colors import LinearSegmentedColormap
from skimage import data
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import (
    QLabel,
    QSizePolicy,
    QScrollArea,
    QMessageBox,
    QMainWindow,
    QMenu,
    QAction,
    qApp,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from color_proc import *
from make_mask import *
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorboard as tb
import subprocess as sub

class MainWindow(QtWidgets.QMainWindow):
    errorSignal = QtCore.pyqtSignal(str)
    outputSignal = QtCore.pyqtSignal(str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("form.ui", self)
        self.scrollArea.setStyleSheet(
            'background-image:url("./icon/tti.png"); background-position: center;'
        )
        self.scrollleft.setStyleSheet(
            'background-image:url("./icon/tti.png"); background-position: center;'
        )
        self.printer = QPrinter()
        self.FileName = None
        self.cvImgData = None
        self.scaleFactor = 1.0
        self.roi = None
        self.gd, self.guidedDAB = None, None
        self.pi_fr, self.pi_os, self.pi_cs, self.pi_ot1, self.pi_ot2 = 3, 5, 5, 0, 255

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        #        self.scrollArea.setVisible(False)
        # get coordinates
        self.piplot = MplCanvas(self)

        self.imageLabel_left = QLabel()
        self.imageLabel_left.setBackgroundRole(QPalette.Base)
        self.imageLabel_left.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel_left.setScaledContents(True)

        self.scrollleft.setBackgroundRole(QPalette.Dark)
        self.scrollleft.setWidget(self.imageLabel_left)
        self.selection = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        #        self.scrollleft.setVisible(False)
        self.createActions()
        self.updateTrainingParams()
        self.initTrainingUI()
        self.initPredictUI()
        self.createMenus()
        self.initViewerUI()
        self.initPreprocessUI()
        self.initCalibUI()
        print(self.parseMatText(str(self.textEdit_pil.toPlainText())))

        self.setWindowTitle("Yamalab WSI Toolbox")
        self.setWindowIcon(QtGui.QIcon("./icon/doc256.png"))
        # self.setWindowTitle("山口研-染色変換工具箱")

    def mousePressEvent(self, event):
        # https://stackoverflow.com/questions/50814090/my-qrubberband-coordinates-doesnt-correspond-with-the-ones-of-my-image
        if event.button() == QtCore.Qt.LeftButton:
            position = QtCore.QPoint(event.pos())
            # IF VIEWER FOR PREPROCESS IS OPENED
            # if position in scroll1
            px = position.x()
            py = position.y()
            s1 = {
                "x0": self.scrollArea.x(),
                "y0": self.scrollArea.y(),
                "x1": self.scrollArea.x() + self.scrollArea.width(),
                "y1": self.scrollArea.y() + self.scrollArea.height(),
            }
            # if position in scroll2
            s2 = {
                "x0": self.scrollleft.x(),
                "y0": self.scrollleft.y(),
                "x1": self.scrollleft.x() + self.scrollleft.width(),
                "y1": self.scrollleft.y() + self.scrollleft.height(),
            }
            print(s1, s2)
            if px >= s1["x0"] and py >= s1["y0"] and px <= s1["x1"] and py <= s1["y1"]:
                # in scroll area 1
                refx = self.imageLabel.x() + s1["x0"]
                refy = self.imageLabel.y() + s1["y0"]
                flag = 1
            elif (
                px >= s2["x0"] and py >= s2["y0"] and px <= s2["x1"] and py <= s2["y1"]
            ):
                # in scroll area 2
                refx = self.imageLabel_left.x() + s2["x0"]
                refy = self.imageLabel_left.y() + s2["y0"]
                flag = 2
            else:
                return
            print("p:%s, %s" % (position.x(), position.y()))
            print(refx, refy)
            if self.selection.isVisible():
                # visible selection
                if (self.upper_left - position).manhattanLength() < 20:
                    # close to upper left corner, drag it
                    self.mode = "drag_upper_left"
                elif (self.lower_right - position).manhattanLength() < 20:
                    # close to lower right corner, drag it
                    self.mode = "drag_lower_right"
                else:
                    # clicked somewhere else, hide selection
                    self.selection.hide()
                print("%s, %s" % (self.upper_left.x(), self.upper_left.y()))
                print("%s, %s" % (self.lower_right.x(), self.lower_right.y()))

                realpos = position - self.imageLabel_left.pos()
                roi = np.array(
                    [
                        self.upper_left.x() - refx,
                        self.upper_left.y() - refy,
                        self.lower_right.x() - refx,
                        self.lower_right.y() - refy,
                    ]
                )
                if roi[3] - roi[1] <= 0 or roi[2] - roi[0] <= 0:
                    return
                roi = roi / self.scaleFactor
                print(flag, roi)
                if flag == 1:
                    self.roicoordH = roi # image coord
                    roi = self.cvImgDataH[
                        int(roi[1]) : int(roi[3]), int(roi[0]) : int(roi[2]), :
                    ]
                    self.roiH = roi
                else:
                    self.roicoordI = roi
                    roi = self.cvImgDataI[
                        int(roi[1]) : int(roi[3]), int(roi[0]) : int(roi[2]), :
                    ]
                    self.roiI = roi

                    self.updatePlotPI()
            else:
                # no visible selection, start new selection
                self.upper_left = position
                self.lower_right = position
                self.mode = "drag_lower_right"
                self.selection.show()

    def updatePlotPI(self):
        if self.roiI is not None:
            self.gd, self.guidedDAB = genIHCMask(
                self.roiI,
                Hinv,
                t=self.horizontalSlider_6.value() / 255,
                grad=self.horizontalSlider.value(),
                MOP_SIZE=self.horizontalSlider_2.value(),
                MCL_SIZE=self.horizontalSlider_3.value(),
                t0=self.horizontalSlider_4.value(),
                t1=self.horizontalSlider_5.value(),
            )
            self.piplot = MplCanvas(self)
            self.piplot.ax1.imshow(self.gd, cmap="gray")
            self.piplot.ax1.axis("off")
            self.piplot.ax1.set_title("DAB")
            self.piplot.ax2.imshow(self.guidedDAB, cmap="gray")
            self.piplot.ax2.axis("off")
            self.piplot.ax2.set_title("Mask")
            self.scrollArea_pi_plot.setWidget(self.piplot)

    def mouseMoveEvent(self, event):
        if self.selection.isVisible():
            # visible selection
            if self.mode == "drag_lower_right":
                self.lower_right = QtCore.QPoint(event.pos())
            elif self.mode == "drag_upper_left":
                self.upper_left = QtCore.QPoint(event.pos())
            # update geometry
            self.selection.setGeometry(
                QtCore.QRect(self.upper_left, self.lower_right).normalized()
            )

    def open(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Images (*.png *.jpeg *.jpg *.bmp *.gif *.tif)",
            options=options,
        )
        self.filename = fileName
        if fileName:
            image = QImage(fileName)
            self.cvImgData = cv.imread(self.filename, cv.CV_32F)
            self.cvImgData = cv.cvtColor(self.cvImgData, cv.COLOR_BGR2RGB) / 255.0

            if image.isNull():
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % fileName
                )
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))

            self.scrollArea.setVisible(True)

            self.fitToWindowAct.setEnabled(True)
            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def openHE(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Images (*.png *.jpeg *.jpg *.bmp *.gif *.tif)",
            options=options,
        )
        self.filename = fileName
        if fileName:
            image = QImage(fileName)
            self.cvImgDataH = cv.imread(self.filename, cv.CV_32F)
            self.cvImgDataH = cv.cvtColor(self.cvImgDataH, cv.COLOR_BGR2RGB)

            if image.isNull():
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % fileName
                )
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))

            self.scrollArea.setVisible(True)

            self.fitToWindowAct.setEnabled(True)
            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def openIHC(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Images (*.png *.jpeg *.jpg *.bmp *.gif *.tif)",
            options=options,
        )
        self.filename = fileName
        if fileName:
            image = QImage(fileName)
            self.cvImgDataI = cv.imread(self.filename, cv.CV_32F)
            self.cvImgDataI = cv.cvtColor(self.cvImgDataI, cv.COLOR_BGR2RGB) / 255
            print(self.cvImgDataI.shape)

            if image.isNull():
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % fileName
                )
                return

            self.imageLabel_left.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0
            self.scrollleft.setVisible(True)

            self.fitToWindowAct.setEnabled(True)
            if not self.fitToWindowAct.isChecked():
                self.imageLabel_left.adjustSize()

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.imageLabel_left.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        self.scrollleft.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()
        self.updateActions()

    def syncViewers(self):
        if self.checkBox_viewlock.isChecked():
            self.imageLabel.move(self.imageLabel_left.pos())

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction(
            "Zoom &In (25%)",
            self,
            shortcut="Ctrl++",
            enabled=True,
            triggered=self.zoomIn,
        )
        self.zoomOutAct = QAction(
            "Zoom &Out (25%)",
            self,
            shortcut="Ctrl+-",
            enabled=True,
            triggered=self.zoomOut,
        )
        self.normalSizeAct = QAction(
            "&Normal Size",
            self,
            shortcut="Ctrl+S",
            enabled=False,
            triggered=self.normalSize,
        )
        self.fitToWindowAct = QAction(
            "&Fit to Window",
            self,
            enabled=False,
            checkable=True,
            shortcut="Ctrl+F",
            triggered=self.fitToWindow,
        )
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

# Training toolbox

    def updateTrainingParams(self):
        self.homepath = self.lineEdit_7.text()
        self.devices = self.lineEdit_2.text()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.devices
        self.learning_rate = self.lineEdit_9.text()
        self.input_size = self.lineEdit_4.text()
        self.test_size = self.lineEdit_8.text()
        self.num_epochs = self.lineEdit_5.text()
        self.batch_size = self.lineEdit_3.text()
        self.model_name = self.lineEdit.text()
        self.loss_name = str(self.comboBox.currentText())
        self.opt_name = str(self.comboBox_2.currentText())

    def initTrainingUI(self):
        self.lossplotLabel = QLabel()
        self.scrollArea_3.setBackgroundRole(QPalette.Dark)
        self.scrollArea_3.setWidget(self.lossplotLabel)
        self.trainExecButton.clicked.connect(self.execTraining)
        self.lineEdit_7.textChanged.connect(self.updateTrainingParams)
        self.lineEdit.textChanged.connect(self.updateTrainingParams)
        self.lineEdit_2.textChanged.connect(self.updateTrainingParams)
        self.lineEdit_9.textChanged.connect(self.updateTrainingParams)
        self.lineEdit_4.textChanged.connect(self.updateTrainingParams)
        self.lineEdit_8.textChanged.connect(self.updateTrainingParams)
        self.lineEdit_5.textChanged.connect(self.updateTrainingParams)
        self.lineEdit_3.textChanged.connect(self.updateTrainingParams)
        self.pushButton_17.clicked.connect(self.launchTensorboard)
        self.pushButton_18.clicked.connect(self.terminateTraining)

        self.webwin = QWebEngineView()
        self.scrollArea_3.setWidget(self.webwin)
        self.webwin.loadFinished.connect(self.adjustTBsize)
    def adjustTBsize(self):
        frame = self.webwin.page().setZoomFactor(0.2)

    def terminateTraining(self):
        self.pt.terminate()
        self.label_31.setText("Terminated")
        self.textEdit_4.append("Terminated\n")

    def launchTensorboard(self):
        port = 6007
        print("tensorboard lanuched on port %s"%port)
        self.port=port
        self.ptb = QtCore.QProcess()
        self.ptb.start('bash -c "source /home/cunyuan/anaconda3/bin/activate tf2 && tensorboard --port=%s --logdir %s/logs/scalars"'%(self.port, self.homepath))
        self.webwin.setUrl(QtCore.QUrl("http://127.0.0.1:%s/#scalars"%port))
        self.webwin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def onReadyReadStandardError(self):
        error = self.pt.readAllStandardError().data().decode()
        self.textEdit_4.append(error)
        self.errorSignal.emit(error)

    def onReadyReadStandardOutput(self):
        result = self.pt.readAllStandardOutput().data().decode()
        self.textEdit_4.append(result)
        self.outputSignal.emit(result)

    def execTraining(self):
        self.label_31.setText("Processing")
        self.pt = QtCore.QProcess()
        self.pt.readyReadStandardError.connect(self.onReadyReadStandardError)
        self.pt.readyReadStandardOutput.connect(self.onReadyReadStandardOutput)

        self.pt.start('bash -c "source /home/cunyuan/anaconda3/bin/activate tf2&& python unet-ki67/main.py -hp %s -d %s -lr %s -is %s -ts %s -e %s -bs %s -mn %s -ln %s"'%(self.homepath, self.devices, self.learning_rate, self. input_size, self.test_size, self.num_epochs, self.batch_size, self.model_name, self.loss_name))

    def getLoggedScalars(self):
        logpath = self.homepath + "/logs/scalars/*"
        list_of_files = glob.glob(logpath)
        latest_file = max(list_of_files, key=os.path.getctime)




# Prediction toolbox
    def releaseNetSpace(self):
        del self.net

    def openPredModel(self):
        dialog = QFileDialog()
        self.model_name =  dialog.getExistingDirectory(self, 'Select model directory')
        self.textEdit_8.append("Loading GPU computing assets...")
        self.textEdit_8.append("Tensorflow and pretrained model successfully loaded.")
        self.textEdit_5.setText(self.model_name)
        self.net = load_model(self.model_name, compile=False)

    def initPredictUI(self):
        self.preproc_interactive_pushButton_browse_5.clicked.connect(self.openPredModel)
        self.preproc_interactive_pushButton_browse_HE_3.clicked.connect(self.openHE)
        self.pushButton_31.clicked.connect(self.predROI)
        self.pushButton_32.clicked.connect(self.releaseNetSpace)

    def predROI(self):
        # print(self.cvImgDataH/255.)
        self.resROI = interactive_prediction(self.roiH/255., self.net)
        # print(resROI)
        superimg = self.cvImgDataH.copy()
        superimg[int(self.roicoordH[1]):int(self.roicoordH[3]),int(self.roicoordH[0]):int(self.roicoordH[2]),:] = self.resROI*255
        self.prplot = MplCanvas1(self)
        self.prplot.ax1.imshow(superimg)
        self.prplot.ax1.axis("off")
        self.scrollArea_pred.setWidget(self.prplot)


# Color calib toolbox

    def initCalibUI(self):
        self.preproc_interactive_pushButton_browse_2.clicked.connect(self.openHE)#IHC
        self.pushButton_15.clicked.connect(self.clearSamples)
        self.pushButton_14.clicked.connect(self.addSampleLabel)
        self.pushButton_33.clicked.connect(self.calibColor)
        self.numSamples = 0
        self.refSampleList = []
        self.calibSampleList = []
        self.labelSampleList=[]
        self.labelSampleListI=[]

    def calibColor(self):
        c011 = np.array([0.,0.,0.])
        Hinv = np.linalg.inv(norm_by_row(self.parseMatText(str(self.textEdit_calib.toPlainText()))))
        k=0
        h_amt, e_amt, d_amt = 0.,0.,0.
        rh_amt, re_amt, rd_amt = 0.,0.,0.
        for im_ref, im_res in zip(self.refSampleList, self.calibSampleList):
            hed_ref = abs(rgbdeconv(im_ref, Hinv))
            hed_res = abs(rgbdeconv(im_res, Hinv))
            h0, w0 = hed_ref.shape[0], hed_ref.shape[1]
            h1, w1 = hed_res.shape[0], hed_res.shape[1]

            h_amt = h_amt + hed_ref[:,:,0].sum()*h1*w1
            rh_amt = rh_amt + hed_res[:,:,0].sum()*h0*w0
            e_amt = e_amt + hed_ref[:,:,1].sum()*h1*w1
            re_amt = re_amt + hed_res[:,:,1].sum()*h0*w0
            d_amt = d_amt + hed_ref[:,:,2].sum()*h1*w1
            rd_amt = rd_amt + hed_res[:,:,2].sum()*h0*w0
        c011 = np.array([h_amt/rh_amt,
                        e_amt/re_amt,
                        d_amt/rd_amt])
        print(c011)
        self.doubleSpinBox_16.setValue(c011[0])
        self.doubleSpinBox_14.setValue(c011[1])
        self.doubleSpinBox_15.setValue(c011[2])
        self.c011 = c011
        self.calibDisp()
        return c011

    def calibDisp(self):
        zdh = abs(rgbdeconv(self.cvImgDataI, Hinv))
        zdh = (zdh.reshape(-1,3)*self.c011.T).reshape(zdh.shape)
        correct_zdh = hecconv(zdh, H)
        correct_zdh = rescale_intensity(correct_zdh)
        self.imageLabel_left.setPixmap(QPixmap.fromImage(QImage((correct_zdh*255).astype(uint8).data, correct_zdh.shape[0], correct_zdh.shape[1], 3*correct_zdh.shape[0], QImage.Format_RGB888)))

    def calibColorMasked(self):
        pass

    def calibColorMurakamiPyramid(self):
        pass

    def addSampleLabel(self):
        self.numSamples += 1
        self.labelSampleList.append(MplCanvas1())
        self.labelSampleList[-1].ax1.imshow(self.roiH)
        self.labelSampleList[-1].ax1.axis(False)
        self.layoutRef.addWidget(self.labelSampleList[-1])

        self.labelSampleListI.append(MplCanvas1())
        self.labelSampleListI[-1].ax1.imshow(self.roiI)
        self.labelSampleListI[-1].ax1.axis(False)
        self.layoutCalib.addWidget(self.labelSampleListI[-1])

        self.refSampleList.append(self.roiH/255.)
        self.calibSampleList.append(self.roiI)


    def clearSamples(self):
        self.numSamples = 0
        self.refSampleList = []
        self.calibSampleList = []
        self.labelSampleList=[]
        self.labelSampleListI=[]
        for i in reversed(range(self.layoutCalib.count())):
            self.layoutCalib.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.layoutRef.count())):
            self.layoutRef.itemAt(i).widget().setParent(None)
#Misc
    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())
        self.imageLabel_left.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)
        self.adjustScrollBar(self.scrollleft.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollleft.verticalScrollBar(), factor)
        self.zoomInAct.setEnabled(self.scaleFactor < 10.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.033)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(
            int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2))
        )

    def initViewerUI(self):
        self.preproc_interactive_pushButton_browse.clicked.connect(self.openHE)
        self.preproc_interactive_pushButton_browse_IHC.clicked.connect(self.openIHC)
        self.scrollArea.horizontalScrollBar().sliderMoved.connect(self.syncViewers)
        self.scrollArea.verticalScrollBar().sliderMoved.connect(self.syncViewers)
        self.scrollleft.horizontalScrollBar().sliderMoved.connect(self.syncViewers)
        self.scrollleft.verticalScrollBar().sliderMoved.connect(self.syncViewers)

    def initPreprocessUI(self):
        self.horizontalSlider.setTickInterval(1)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setMinimum(1)
        self.horizontalSlider.setMaximum(15)
        self.horizontalSlider.setValue(3)
        self.horizontalSlider_2.setTickInterval(1)
        self.horizontalSlider_2.setSingleStep(1)
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_2.setMaximum(20)
        self.horizontalSlider_2.setValue(5)
        self.horizontalSlider_3.setTickInterval(1)
        self.horizontalSlider_3.setSingleStep(1)
        self.horizontalSlider_3.setMinimum(0)
        self.horizontalSlider_3.setMaximum(20)
        self.horizontalSlider_3.setValue(5)
        self.horizontalSlider_4.setTickInterval(10)
        self.horizontalSlider_4.setSingleStep(1)
        self.horizontalSlider_4.setMinimum(0)
        self.horizontalSlider_4.setMaximum(255)
        self.horizontalSlider_4.setValue(0)
        self.horizontalSlider_5.setTickInterval(10)
        self.horizontalSlider_5.setSingleStep(1)
        self.horizontalSlider_5.setMinimum(0)
        self.horizontalSlider_5.setMaximum(255)
        self.horizontalSlider_5.setValue(255)
        self.horizontalSlider_6.setTickInterval(10)
        self.horizontalSlider_6.setSingleStep(1)
        self.horizontalSlider_6.setMinimum(0)
        self.horizontalSlider_6.setMaximum(255)
        self.horizontalSlider_6.setValue(0.2 * 255 // 1)

        self.doubleSpinBox.setSingleStep(1)
        self.doubleSpinBox.setMinimum(1)
        self.doubleSpinBox.setMaximum(15)
        self.doubleSpinBox.setValue(3)
        self.doubleSpinBox_2.setSingleStep(1)
        self.doubleSpinBox_2.setMinimum(0)
        self.doubleSpinBox_2.setMaximum(20)
        self.doubleSpinBox_2.setValue(5)
        self.doubleSpinBox_3.setSingleStep(1)
        self.doubleSpinBox_3.setMinimum(0)
        self.doubleSpinBox_3.setMaximum(20)
        self.doubleSpinBox_3.setValue(5)
        self.doubleSpinBox_4.setSingleStep(1)
        self.doubleSpinBox_4.setMinimum(0)
        self.doubleSpinBox_4.setMaximum(255)
        self.doubleSpinBox_4.setValue(0)
        self.doubleSpinBox_5.setSingleStep(1)
        self.doubleSpinBox_5.setMinimum(0)
        self.doubleSpinBox_5.setMaximum(255)
        self.doubleSpinBox_5.setValue(255)
        self.doubleSpinBox_6.setSingleStep(1)
        self.doubleSpinBox_6.setMinimum(0)
        self.doubleSpinBox_6.setMaximum(255)
        self.doubleSpinBox_6.setValue(255 * 0.2 // 1)

        self.horizontalSlider.valueChanged.connect(self.doubleSpinBox.setValue)
        self.doubleSpinBox.valueChanged.connect(self.horizontalSlider.setValue)
        self.horizontalSlider_2.valueChanged.connect(self.doubleSpinBox_2.setValue)
        self.doubleSpinBox_2.valueChanged.connect(self.horizontalSlider_2.setValue)
        self.horizontalSlider_3.valueChanged.connect(self.doubleSpinBox_3.setValue)
        self.doubleSpinBox_3.valueChanged.connect(self.horizontalSlider_3.setValue)
        self.horizontalSlider_4.valueChanged.connect(self.doubleSpinBox_4.setValue)
        self.doubleSpinBox_4.valueChanged.connect(self.horizontalSlider_4.setValue)
        self.horizontalSlider_5.valueChanged.connect(self.doubleSpinBox_5.setValue)
        self.doubleSpinBox_5.valueChanged.connect(self.horizontalSlider_5.setValue)
        self.horizontalSlider_6.valueChanged.connect(self.doubleSpinBox_6.setValue)
        self.doubleSpinBox_6.valueChanged.connect(self.horizontalSlider_6.setValue)

        self.horizontalSlider.valueChanged.connect(self.updatePlotPI)
        self.horizontalSlider_2.valueChanged.connect(self.updatePlotPI)
        self.horizontalSlider_3.valueChanged.connect(self.updatePlotPI)
        self.horizontalSlider_4.valueChanged.connect(self.updatePlotPI)
        self.horizontalSlider_5.valueChanged.connect(self.updatePlotPI)
        self.horizontalSlider_6.valueChanged.connect(self.updatePlotPI)

    def parseMatText(self, datastring):
        df = pd.read_csv(io.StringIO(datastring), header=None, sep=",").to_numpy()
        print(df)
        return df

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = fig.add_subplot(121)
        self.ax2 = fig.add_subplot(122)
        super(MplCanvas, self).__init__(fig)

class MplCanvas1(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=10, dpi=300):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = fig.add_subplot()
        super(MplCanvas1, self).__init__(fig)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
