from qtpy.QtWidgets import QVBoxLayout, QRadioButton, QLabel, QPushButton, QDialog
from qtpy.QtCore import Qt
from labelme.label_file import LabelFile, LabelFileFormat

class FormatSelectionDialog(QDialog):
    def __init__(self):
        super(FormatSelectionDialog, self).__init__()

        # Set the window title
        self.setWindowTitle("Select Format")

        # Set the layout of the widget
        layout = QVBoxLayout()

        # Create a label asking the question
        question_label = QLabel("Which format do you want to work with?")
        layout.addWidget(question_label, alignment=Qt.AlignCenter)

        # Create radio buttons for JSON and YOLO
        self.json_radio = QRadioButton("JSON")
        self.yolo_radio = QRadioButton("YOLO")

        # Add the radio buttons to the layout
        layout.addWidget(self.json_radio)
        layout.addWidget(self.yolo_radio)

        # Create a button to continue
        continue_button = QPushButton("Continue")
        continue_button.clicked.connect(self.accept)

        # Add the button at the end of the layout
        layout.addWidget(continue_button, alignment=Qt.AlignCenter)

        # Default to current label file type
        if LabelFile.suffix == ".json":
            self.json_radio.setChecked(True)
        else:
            self.yolo_radio.setChecked(True)

        # Set the layout to the widget
        self.setLayout(layout)

    def selected_format(self):
        return LabelFileFormat.JSON if self.json_radio.isChecked() else LabelFileFormat.YOLO
