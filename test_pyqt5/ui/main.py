# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Main(object):
    def setupUi(self, Main):
        Main.setObjectName("Main")
        Main.resize(400, 161)
        self.widget = QtWidgets.QWidget(Main)
        self.widget.setGeometry(QtCore.QRect(20, 10, 361, 141))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QtWidgets.QGroupBox(self.widget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radioBtnThree = QtWidgets.QRadioButton(self.groupBox)
        self.radioBtnThree.setObjectName("radioBtnThree")
        self.horizontalLayout.addWidget(self.radioBtnThree)
        self.radioBtnTwo = QtWidgets.QRadioButton(self.groupBox)
        self.radioBtnTwo.setObjectName("radioBtnTwo")
        self.horizontalLayout.addWidget(self.radioBtnTwo)
        self.radioBtnMis_Tar = QtWidgets.QRadioButton(self.groupBox)
        self.radioBtnMis_Tar.setObjectName("radioBtnMis_Tar")
        self.horizontalLayout.addWidget(self.radioBtnMis_Tar)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.pushBtnRun = QtWidgets.QPushButton(self.widget)
        self.pushBtnRun.setObjectName("pushBtnRun")
        self.horizontalLayout_2.addWidget(self.pushBtnRun)
        self.pushBtnClose = QtWidgets.QPushButton(self.widget)
        self.pushBtnClose.setObjectName("pushBtnClose")
        self.horizontalLayout_2.addWidget(self.pushBtnClose)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        self.retranslateUi(Main)
        QtCore.QMetaObject.connectSlotsByName(Main)

    def retranslateUi(self, Main):
        _translate = QtCore.QCoreApplication.translate
        Main.setWindowTitle(_translate("Main", "Form"))
        self.groupBox.setTitle(_translate("Main", "绘图方式选择"))
        self.radioBtnThree.setText(_translate("Main", "三维图"))
        self.radioBtnTwo.setText(_translate("Main", "二维图"))
        self.radioBtnMis_Tar.setText(_translate("Main", "弹目交互"))
        self.pushBtnRun.setText(_translate("Main", "运行"))
        self.pushBtnClose.setText(_translate("Main", "关闭"))
