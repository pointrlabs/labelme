from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class SnakeWidget(QtWidgets.QSpinBox):
    def __init__(self, value=2):
        super(SnakeWidget, self).__init__()
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.setRange(1, 32)
        self.setPrefix("1/")
        self.setValue(value)
        self.setToolTip("Snake DownsamplingRatio")
        self.setStatusTip(self.toolTip())
        self.setAlignment(QtCore.Qt.AlignCenter)

    def minimumSizeHint(self):
        height = super(SnakeWidget, self).minimumSizeHint().height()
        fm = QtGui.QFontMetrics(self.font())
        width = fm.width(str(self.maximum()))
        return QtCore.QSize(width, height)
