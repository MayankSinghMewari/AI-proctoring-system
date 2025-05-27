from tkinter import *
from tkinter import ttk

class ProctoringUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Proctoring System")
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid()
        
        # Add notification label
        self.notification_label = Label(self.frm, text="", fg="red", font=("Arial", 12))
        self.notification_label.grid(row=0, column=0, pady=10)
        
        # Add status label
        self.status_label = Label(self.frm, text="Monitoring...", fg="green", font=("Arial", 12))
        self.status_label.grid(row=1, column=0, pady=10)

    def show_notification(self, message):
        self.root.after(0, self.notification_label.config, {"text": message}) 
        self.notification_label.config(text=message)
        self.root.update()

    def run(self):
        self.root.mainloop()

# Create a global instance of the UI
ui = ProctoringUI()