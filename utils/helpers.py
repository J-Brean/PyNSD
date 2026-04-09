from PyQt6.QtWidgets import QLabel                                     # Import label widget
from PyQt6.QtCore import Qt                                            # Import Qt enumerations

def create_info_icon(tooltip_text: str) -> QLabel:                     # Helper to create info emojis
    lbl = QLabel("ℹ️")                                                 # Use an information emoji
    lbl.setToolTip(tooltip_text)                                       # Attach the descriptive text
    lbl.setStyleSheet("font-size: 16px; margin-left: 5px;")            # Make it slightly larger
    lbl.setCursor(Qt.CursorShape.WhatsThisCursor)                      # Change mouse to a question mark on hover
    return lbl                                                         # Return the ready-to-use widget