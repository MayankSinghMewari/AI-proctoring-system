from unicodedata import name
import wmi
import time
start= time.time()
# Initializing the wmi constructor
f = wmi.WMI()

notAllowed = [
    "Discord",
    "Whatsapp",
    "Telegram",
    "Zoom",
    "Skype",
    "BlueStacks", 
    "NoxPlayer"
]
print("Name         Id")
x = f.Win32_Process()
cheating_detected = False
for process in x:
    for name in notAllowed:
        if name.lower() in process.Name.lower():
            print(process.Name,process.ProcessId)
            cheating_detected = True

            
if cheating_detected:
    print("Cheating detected: User is running disallowed applications.")
else:
    print("No disallowed applications found.")

print("Found {} processes".format(len(x)))
end = time.time()
print("Executed in {} seconds".format(end-start))