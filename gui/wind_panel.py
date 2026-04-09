from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class WindPanel(QWidget):
    def __init__(self, main_df=None, diams=None, parent=None):
        super().__init__(parent)
        
        # Placeholders for when you eventually pass data into this panel
        self.main_df = main_df
        self.diams = diams
        
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # A friendly placeholder so you know it loaded!
        title = QLabel("<h2>Wind Direction Analysis Panel</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("<i>This panel is currently under construction. Stay tuned!</i>")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        # Pushes your placeholders to the top-center
        layout.addStretch()
        
    def load_data(self, data_file):
        """Standard method to catch data when the user loads a file in the main window."""
        self.df = data_file.df
        self.diams = data_file.diameters
        # Add your future PMF load logic here!