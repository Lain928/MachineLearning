# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mis_tar.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mis_tar(object):
    def setupUi(self, mis_tar):
        mis_tar.setObjectName("mis_tar")
        mis_tar.resize(565, 167)
        self.layoutWidget = QtWidgets.QWidget(mis_tar)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 0, 501, 141))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit_mis = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_mis.setObjectName("lineEdit_mis")
        self.horizontalLayout.addWidget(self.lineEdit_mis)
        self.btn_mis_choose = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_mis_choose.setObjectName("btn_mis_choose")
        self.horizontalLayout.addWidget(self.btn_mis_choose)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_mis_run = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_mis_run.setObjectName("btn_mis_run")
        self.horizontalLayout_2.addWidget(self.btn_mis_run)
        self.btn_mis_close = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_mis_close.setObjectName("btn_mis_close")
        self.horizontalLayout_2.addWidget(self.btn_mis_close)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(mis_tar)
        QtCore.QMetaObject.connectSlotsByName(mis_tar)

    def retranslateUi(self, mis_tar):
        _translate = QtCore.QCoreApplication.translate
        mis_tar.setWindowTitle(_translate("mis_tar", "Form"))
        self.label.setText(_translate("mis_tar", "弹目交互数据绘图"))
        self.btn_mis_choose.setText(_translate("mis_tar", "选择文件"))
        self.btn_mis_run.setText(_translate("mis_tar", "运行"))
        self.btn_mis_close.setText(_translate("mis_tar", "退出"))