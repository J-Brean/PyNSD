import sys

from PyQt6.QtWidgets import QApplication

from gui.theme import apply_app_theme, make_splash

if __name__ == "__main__":
    # 1. Start the application engine and install the shared theme
    app = QApplication(sys.argv)
    apply_app_theme(app)

    # 2. Show a proper splash screen while the heavy packages import
    splash = make_splash()
    splash.show()
    app.processEvents()

    # 3. NOW load the heavy packages (pandas/numpy/matplotlib via the panels)
    from gui.main_window import MainWindow

    # 4. Build the main app
    window = MainWindow()

    # 5. Hand off from splash to the real window
    splash.finish(window)
    window.show()

    sys.exit(app.exec())
