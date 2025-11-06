# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\console_UI.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(182, 63)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 182, 21))
        self.menubar.setObjectName("menubar")
        self.menurestart = QtWidgets.QMenu(self.menubar)
        self.menurestart.setObjectName("menurestart")
        self.menuclose = QtWidgets.QMenu(self.menubar)
        self.menuclose.setObjectName("menuclose")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionView = QtWidgets.QAction(MainWindow)
        self.actionView.setObjectName("actionView")
        self.actionmsi = QtWidgets.QAction(MainWindow)
        self.actionmsi.setObjectName("actionmsi")
        self.actionMSI = QtWidgets.QAction(MainWindow)
        self.actionMSI.setObjectName("actionMSI")
        self.actionView_2 = QtWidgets.QAction(MainWindow)
        self.actionView_2.setObjectName("actionView_2")
        self.actionMSI_2 = QtWidgets.QAction(MainWindow)
        self.actionMSI_2.setObjectName("actionMSI_2")
        self.menurestart.addAction(self.actionView)
        self.menurestart.addAction(self.actionMSI)
        self.menuclose.addAction(self.actionView_2)
        self.menuclose.addAction(self.actionMSI_2)
        self.menubar.addAction(self.menurestart.menuAction())
        self.menubar.addAction(self.menuclose.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Console"))
        self.menurestart.setTitle(_translate("MainWindow", "restart"))
        self.menuclose.setTitle(_translate("MainWindow", "close"))
        self.actionView.setText(_translate("MainWindow", "View"))
        self.actionmsi.setText(_translate("MainWindow", "msi"))
        self.actionMSI.setText(_translate("MainWindow", "MSI"))
        self.actionView_2.setText(_translate("MainWindow", "View"))
        self.actionMSI_2.setText(_translate("MainWindow", "MSI"))


