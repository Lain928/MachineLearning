# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'threedim.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_three_dim(object):
    def setupUi(self, three_dim):
        three_dim.setObjectName("three_dim")
        three_dim.resize(508, 163)
        self.widget = QtWidgets.QWidget(three_dim)
        self.widget.setGeometry(QtCore.QRect(30, 0, 451, 151))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.Box_three_x = QtWidgets.QComboBox(self.widget)
        self.Box_three_x.setObjectName("Box_three_x")
        self.horizontalLayout.addWidget(self.Box_three_x)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.Box_three_y = QtWidgets.QComboBox(self.widget)
        self.Box_three_y.setObjectName("Box_three_y")
        self.horizontalLayout.addWidget(self.Box_three_y)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.Box_three_z = QtWidgets.QComboBox(self.widget)
        self.Box_three_z.setObjectName("Box_three_z")
        self.horizontalLayout.addWidget(self.Box_three_z)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_three = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_three.setObjectName("lineEdit_three")
        self.horizontalLayout_3.addWidget(self.lineEdit_three)
        self.btn_three_choose = QtWidgets.QPushButton(self.widget)
        self.btn_three_choose.setObjectName("btn_three_choose")
        self.horizontalLayout_3.addWidget(self.btn_three_choose)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_three_run = QtWidgets.QPushButton(self.widget)
        self.btn_three_run.setObjectName("btn_three_run")
        self.horizontalLayout_2.addWidget(self.btn_three_run)
        self.btn_three_close = QtWidgets.QPushButton(self.widget)
        self.btn_three_close.setObjectName("btn_three_close")
        self.horizontalLayout_2.addWidget(self.btn_three_close)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(three_dim)
        QtCore.QMetaObject.connectSlotsByName(three_dim)

    def retranslateUi(self, three_dim):
        _translate = QtCore.QCoreApplication.translate
        three_dim.setWindowTitle(_translate("three_dim", "Form"))
        self.label.setText(_translate("three_dim", "x:"))
        self.label_2.setText(_translate("three_dim", "y:"))
        self.label_3.setText(_translate("three_dim", "z:"))
        self.btn_three_choose.setText(_translate("three_dim", "选择文件"))
        self.btn_three_run.setText(_translate("three_dim", "运行"))
        self.btn_three_close.setText(_translate("three_dim", "退出"))
