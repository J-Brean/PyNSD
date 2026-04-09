import sys
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtCore import Qt

if __name__ == "__main__":
    # 1. Start the application engine
    app = QApplication(sys.argv)
    
    # 2. Create a custom, borderless loading splash screen
    splash = QLabel("Loading toolkits...")
    splash.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
    splash.setStyleSheet("""
        font-family: Georgia, serif; 
        font-size: 24px; 
        color: #33302e; 
        background-color: #fff1e5; 
        padding: 50px; 
        border: 2px solid #b3a8a0; 
        border-radius: 10px;
    """)
    splash.setAlignment(Qt.AlignmentFlag.AlignCenter)
    splash.show()
    
    # 3. Force the OS to render the splash box right now
    app.processEvents()
    
    # 4. NOW load the heavy packages. The splash screen will stay frozen on screen.
    from gui.main_window import MainWindow

    # 5. Build the main app
    window = MainWindow()
    
    # 6. Destroy the loading screen and show the actual app
    splash.close()
    window.show()
    
    sys.exit(app.exec())