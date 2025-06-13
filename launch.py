# launch.py

import subprocess
import time
import sys
from pyngrok import ngrok

# --- Configuration ---
# The port your Streamlit app will run on. Default is 8501
STREAMLIT_PORT = 8501
# -------------------

# Start the Streamlit app as a separate process
# The command is "streamlit run app.py"
# We add --server.port to ensure it runs on our chosen port
process = subprocess.Popen([
    sys.executable,  # <-- Use the full path to the current python.exe
    "-m",            # <-- Tell python to run a module
    "streamlit",
    "run",
    "app.py",
    "--server.port",
    str(STREAMLIT_PORT)
])

print("🚀 Starting Streamlit app...")
# Give the app a moment to start
time.sleep(5)

# Create a public URL tunnel with ngrok
try:
    public_url = ngrok.connect(STREAMLIT_PORT, "http")
    print("=" * 50)
    print("✅ Your app is live!")
    print(f"🔗 Public URL: {public_url}")
    print("=" * 50)
    print("(Press Ctrl+C in this terminal to stop the app and the tunnel)")

    # Keep the script alive to keep the tunnel open
    process.wait()

except Exception as e:
    print(f"❌ An error occurred: {e}")

finally:
    # This will run when you press Ctrl+C
    print("\n shutting down...")
    ngrok.kill()
    process.terminate()