# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'new_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_UI_New(object):
    def setupUi(self, UI_New):
        UI_New.setObjectName("UI_New")
        UI_New.resize(550, 254)
        self.layoutWidget = QtWidgets.QWidget(UI_New)
        self.layoutWidget.setGeometry(QtCore.QRect(11, 11, 531, 231))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.radio_three = QtWidgets.QRadioButton(self.layoutWidget)
        self.radio_three.setObjectName("radio_three")
        self.horizontalLayout_3.addWidget(self.radio_three)
        self.radio_two = QtWidgets.QRadioButton(self.layoutWidget)
        self.radio_two.setObjectName("radio_two")
        self.horizontalLayout_3.addWidget(self.radio_two)
        self.radio_mis = QtWidgets.QRadioButton(self.layoutWidget)
        self.radio_mis.setObjectName("radio_mis")
        self.horizontalLayout_3.addWidget(self.radio_mis)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_x = QtWidgets.QLabel(self.layoutWidget)
        self.label_x.setObjectName("label_x")
        self.horizontalLayout_5.addWidget(self.label_x)
        self.comboBox_x = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBox_x.setObjectName("comboBox_x")
        self.horizontalLayout_5.addWidget(self.comboBox_x)
        self.label_y = QtWidgets.QLabel(self.layoutWidget)
        self.label_y.setObjectName("label_y")
        self.horizontalLayout_5.addWidget(self.label_y)
        self.comboBox_y = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBox_y.setObjectName("comboBox_y")
        self.horizontalLayout_5.addWidget(self.comboBox_y)
        self.label_z = QtWidgets.QLabel(self.layoutWidget)
        self.label_z.setObjectName("label_z")
        self.horizontalLayout_5.addWidget(self.label_z)
        self.comboBox_z = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBox_z.setObjectName("comboBox_z")
        self.horizontalLayout_5.addWidget(self.comboBox_z)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit_path = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_path.setObjectName("lineEdit_path")
        self.horizontalLayout_2.addWidget(self.lineEdit_path)
        self.btn_choosefile = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_choosefile.setObjectName("btn_choosefile")
        self.horizontalLayout_2.addWidget(self.btn_choosefile)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_run = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_run.setObjectName("btn_run")
        self.horizontalLayout.addWidget(self.btn_run)
        self.btn_close = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_close.setObjectName("btn_close")
        self.horizontalLayout.addWidget(self.btn_close)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(UI_New)
        QtCore.QMetaObject.connectSlotsByName(UI_New)

    def retranslateUi(self, UI_New):
        _translate = QtCore.QCoreApplication.translate
        UI_New.setWindowTitle(_translate("UI_New", "Form"))
        self.label_4.setText(_translate("UI_New", "绘图方式选择："))
        self.radio_three.setText(_translate("UI_New", "三维绘图"))
        self.radio_two.setText(_translate("UI_New", "二维绘图"))
        self.radio_mis.setText(_translate("UI_New", "弹目交互"))
        self.label_x.setText(_translate("UI_New", "x轴："))
        self.label_y.setText(_translate("UI_New", "y轴："))
        self.label_z.setText(_translate("UI_New", "z轴："))
        self.btn_choosefile.setText(_translate("UI_New", "选择文件"))
        self.btn_run.setText(_translate("UI_New", "运行"))
        self.btn_close.setText(_translate("UI_New", "退出"))