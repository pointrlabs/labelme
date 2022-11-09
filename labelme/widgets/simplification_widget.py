from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class SimplificationWidget(QtWidgets.QSpinBox):
    def __init__(self, value=100):
        super(SimplificationWidget, self).__init__()
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.setRange(0, 10000)
        self.setSuffix(" px2")
        self.setValue(value)
        self.setToolTip("Visvalingam Threshold")
        self.setStatusTip(self.toolTip())
        self.setAlignment(QtCore.Qt.AlignCenter)

    def minimumSizeHint(self):
        height = super(SimplificationWidget, self).minimumSizeHint().height()
        fm = QtGui.QFontMetrics(self.font())
        width = fm.width(str(self.maximum()))
        return QtCore.QSize(width, height)
